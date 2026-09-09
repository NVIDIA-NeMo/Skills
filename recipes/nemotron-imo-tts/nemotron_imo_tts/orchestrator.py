# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The per-problem search: generate, verify, stop a checkpoint on acceptance, refine, fall back.

One ``ProblemOrchestrator`` owns one problem. Round 1 fans out every prompt x checkpoint x sample; each valid
proof is verified by the full panel; an accepted proof becomes its checkpoint's finalist and stops that
checkpoint's remaining work. Each checkpoint stops independently when one of its own proofs is accepted;
checkpoints without an accepted proof keep going until the round ends. Without acceptance, the top-ranked pool proofs
are refined by every checkpoint, round after round, up to the budget. The orchestrator is transport-agnostic:
it receives an async ``request`` callable and a ``sink`` for rows, and asks the ``Ledger`` before issuing
anything so relaunches replay persisted work.
"""

import asyncio
import logging
from dataclasses import dataclass, field

from nemotron_imo_tts import prompting, resume, scoring
from nemotron_imo_tts.client import RequestParked, Response
from nemotron_imo_tts.text import (
    is_complete_finish_reason,
    parse_proof,
    parse_verification_score,
    sha256_text,
    strip_think,
)

LOG = logging.getLogger("nemotron_imo_tts.orchestrator")

ROLE_GEN = "gen"
ROLE_VERIFY = "verify"
ROLE_REFINE = "refine"
STATUS_ACCEPTED = "accepted"
STATUS_FALLBACK = "fallback"
STATUS_NO_CANDIDATES = "no_candidates"


@dataclass
class ProofRecord:
    proof: str
    proof_sha256: str
    self_eval: dict
    self_eval_score: float
    round_idx: int
    seed: int
    generation_model_id: str
    generation_prompt_id: str | None
    generation_random_seed: int
    aggregation_trial_idx: int | None
    dep_proof_ids: list
    origins: list
    proof_id: int
    verifier_scores: dict = field(default_factory=dict)
    verdict: scoring.PanelVerdict | None = None
    critiques: list = field(default_factory=list)
    accepted: bool = False

    @property
    def is_evidence(self):
        return self.verdict is not None and self.verdict.complete

    @property
    def meanscore(self):
        return self.verdict.meanscore if self.verdict is not None else 0.0

    def pool_record(self):
        score2ratings = {}
        for critique in self.critiques:
            score2ratings.setdefault(str(critique["score"]), []).append(dict(critique))
        return {
            "proof": self.proof,
            "proof_sha256": self.proof_sha256,
            "proof_id": self.proof_id,
            "round_idx": self.round_idx,
            "seed": self.seed,
            "generation_model_id": self.generation_model_id,
            "generation_prompt_id": self.generation_prompt_id,
            "generation_random_seed": self.generation_random_seed,
            "aggregation_trial_idx": self.aggregation_trial_idx,
            "dep_proof_ids": list(self.dep_proof_ids),
            "origins": list(self.origins),
            "self_eval": self.self_eval,
            "self_eval_score": self.self_eval_score,
            "meanscore": self.meanscore,
            "verifier_means": dict(self.verdict.verifier_means) if self.verdict else {},
            "verifier_scores": {k: list(v) for k, v in self.verifier_scores.items()},
            "num_valid_votes": len(self.verdict.valid_scores) if self.verdict else 0,
            "verification_complete": self.is_evidence,
            "accepted": self.accepted,
            "score2ratings": score2ratings,
        }


@dataclass
class SearchResult:
    status: str
    rounds_run: int
    finalists: list
    num_candidates: int
    num_parked: int
    num_prompt_too_long: int
    pool: list


def _response_fields(resp):
    return {
        "generation": resp.generation,
        "reasoning_content": resp.reasoning_content,
        "finish_reason": resp.finish_reason,
        "usage": resp.usage,
        "num_generated_tokens": resp.num_generated_tokens,
        "attempts": resp.attempts,
        "elapsed_s": resp.elapsed_s,
        "context_budget": dict(resp.context_budget),
    }


class ProblemOrchestrator:
    def __init__(self, cfg, prompts, request, sink, ledger):
        self.cfg = cfg
        self.prompts = prompts
        self.request = request
        self.sink = sink
        self.ledger = ledger
        self.judgments_per_proof = cfg.verifiers[0].judgments_per_proof
        self.proofs = {}
        self._next_proof_id = 1
        self.finalists = {}
        self.stopped = set()
        self._tasks_by_checkpoint = {}
        self._all_tasks = set()
        self.num_parked = 0
        self.num_prompt_too_long = 0

    # ---- task bookkeeping --------------------------------------------------
    def _spawn(self, model_id, coro):
        task = asyncio.create_task(coro)
        self._all_tasks.add(task)
        self._tasks_by_checkpoint.setdefault(model_id, set()).add(task)
        task.add_done_callback(self._all_tasks.discard)
        task.add_done_callback(lambda t, cid=model_id: self._tasks_by_checkpoint[cid].discard(t))
        return task

    async def _cancel_all(self):
        pending = [t for t in self._all_tasks if not t.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _await_batch(self, tasks):
        """Wait for a round's tasks; a fatal error cancels the rest and propagates."""
        if not tasks:
            return
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        error = None
        for task in done:
            if task.cancelled():
                continue
            if task.exception() is not None:
                error = task.exception()
                break
        if error is not None:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            raise error
        await asyncio.gather(*pending, return_exceptions=False)

    def _on_accepted(self, problem, pr):
        pr.accepted = True
        model_id = pr.generation_model_id
        if model_id in self.finalists:
            return  # a completion race; the first accepted proof stays the finalist
        self.finalists[model_id] = pr
        self.stopped.add(model_id)
        LOG.info(
            "[%s] checkpoint %s accepted proof %d (round %d); stopping its remaining work",
            problem["problem_idx"],
            model_id,
            pr.proof_id,
            pr.round_idx,
        )
        me = asyncio.current_task()
        for task in list(self._tasks_by_checkpoint.get(model_id, ())):
            if task is not me and not task.done():
                task.cancel()

    # ---- requests and rows -------------------------------------------------
    async def _do(self, role, spec, messages, seed, identity):
        row = self.ledger.lookup(identity)
        if row is not None:
            return Response(
                generation=row.get("generation", ""),
                finish_reason=row.get("finish_reason"),
                reasoning_content=None,
                usage=None,
                num_generated_tokens=None,
                attempts=0,
                elapsed_s=0.0,
                replayed=True,
            )
        return await self.request(role, spec, messages, seed, identity)

    async def _durable_sink(self, stage, round_idx, row):
        """Persist a row even if cancellation arrives mid-write; replayed rows are already on disk."""
        if row.pop("_replayed", False):
            return
        sink_task = asyncio.create_task(self.sink(stage, round_idx, row))
        try:
            await asyncio.shield(sink_task)
        except asyncio.CancelledError:
            await sink_task
            raise

    @staticmethod
    def _base_row(problem):
        return {
            "problem_idx": problem["problem_idx"],
            "source_name": problem.get("source_name", "unknown"),
            "question": problem["question"],
        }

    async def _record_error(self, problem, role, round_idx, details, **provenance):
        category = details.get("category")
        if category == "prompt_too_long":
            self.num_prompt_too_long += 1
        else:
            self.num_parked += 1
        row = self._base_row(problem)
        row.update({"role": role, "round_idx": round_idx, **provenance, **details})
        await self._durable_sink("errors", round_idx, row)

    async def _record_gen(
        self,
        problem,
        resp,
        *,
        role,
        round_idx,
        seed,
        model,
        request_seed,
        prompt_id,
        trial_idx,
        dep_proof_ids,
        proof,
        self_eval,
        valid,
    ):
        row = self._base_row(problem)
        row.update(_response_fields(resp))
        row.update(
            {
                "role": role,
                "round_idx": round_idx,
                "seed": seed,
                "generation_seed": seed,
                "row_id": f"{problem['problem_idx']}_{seed}",
                "generation_model_id": model.id,
                "generation_random_seed": request_seed,
                "proof": proof,
                "proof_sha256": sha256_text(proof),
                "self_eval": self_eval,
                "self_eval_score": self_eval["self_eval_score"],
                "generation_complete": is_complete_finish_reason(resp.finish_reason),
                "valid": valid,
            }
        )
        if role == ROLE_GEN:
            row["generation_prompt_id"] = prompt_id
        else:
            row["aggregation_trial_idx"] = trial_idx
            row["dep_proof_ids"] = list(dep_proof_ids)
        if resp.replayed:
            row["_replayed"] = True
        await self._durable_sink(role, round_idx, row)

    async def _record_vote(self, problem, pr, verifier, slot, resp, score, counted):
        row = self._base_row(problem)
        row.update(_response_fields(resp))
        rating_text = strip_think(resp.generation)
        row.update(
            {
                "role": ROLE_VERIFY,
                "round_idx": pr.round_idx,
                "proof_sha256": pr.proof_sha256,
                "generation_seed": pr.seed,
                "row_id": f"{problem['problem_idx']}_{pr.seed}",
                "generation_model_id": pr.generation_model_id,
                "generation_prompt_id": pr.generation_prompt_id,
                "verifier_id": verifier.id,
                "verification_seed": slot,
                "verify_row_id": f"{problem['problem_idx']}_{pr.seed}_{verifier.id}_v{slot}",
                "rating_text": rating_text,
                "verification_score": score,
                "verification_complete": is_complete_finish_reason(resp.finish_reason),
                "valid": score is not None,
                "counted_for_evidence": bool(counted),
            }
        )
        if resp.replayed:
            row["_replayed"] = True
        await self._durable_sink(ROLE_VERIFY, pr.round_idx, row)
        return rating_text

    # ---- verification ------------------------------------------------------
    async def _verify_proof(self, problem, pr):
        messages = prompting.user_message(
            prompting.render_verification(self.prompts.verification, problem["question"], pr.proof)
        )
        specs = []
        tasks = []
        for verifier in self.cfg.verifiers:
            for slot in range(self.judgments_per_proof):
                identity = resume.verify_identity(
                    problem["problem_idx"], pr.round_idx, pr.proof_sha256, verifier.id, slot
                )
                specs.append((verifier, slot))
                tasks.append(asyncio.create_task(self._do(ROLE_VERIFY, verifier, messages, slot, identity)))
        interrupted = False
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            interrupted = True
            results = [t.result() if t.done() and not t.cancelled() and t.exception() is None else None for t in tasks]

        scores_by_verifier = {verifier.id: [] for verifier in self.cfg.verifiers}
        received = []
        fatal = None
        for (verifier, slot), result in zip(specs, results):
            if result is None or isinstance(result, asyncio.CancelledError):
                interrupted = True
                continue
            if isinstance(result, RequestParked):
                await self._record_error(
                    problem,
                    ROLE_VERIFY,
                    pr.round_idx,
                    result.details,
                    generation_seed=pr.seed,
                    proof_sha256=pr.proof_sha256,
                    verifier_id=verifier.id,
                    verification_seed=slot,
                )
                continue
            if isinstance(result, BaseException):
                fatal = fatal or result
                continue
            score = (
                parse_verification_score(result.generation)
                if is_complete_finish_reason(result.finish_reason)
                else None
            )
            scores_by_verifier[verifier.id].append(score)
            received.append((verifier, slot, result, score))

        verdict = scoring.panel_verdict(
            scores_by_verifier, self.judgments_per_proof, interrupted=interrupted or fatal is not None
        )
        candidates = []
        for verifier, slot, result, score in received:
            rating_text = await self._record_vote(problem, pr, verifier, slot, result, score, counted=verdict.complete)
            if score is not None:
                candidates.append(
                    {"rating": rating_text, "score": score, "verifier_id": verifier.id, "verification_seed": slot}
                )
        if fatal is not None:
            raise fatal
        if interrupted:
            raise asyncio.CancelledError
        pr.verifier_scores = scores_by_verifier
        pr.verdict = verdict
        pr.critiques = scoring.retain_critiques(candidates, self.cfg.verifier_order) if verdict.complete else []
        LOG.info(
            "[%s] proof %d from %s (round %d): %s -> %s",
            problem["problem_idx"],
            pr.proof_id,
            pr.generation_model_id,
            pr.round_idx,
            " ".join(
                f"{vid}={','.join('-' if s is None else f'{s:g}' for s in scores)}"
                for vid, scores in scores_by_verifier.items()
            ),
            "accepted" if verdict.accepted else ("rejected" if verdict.complete else "incomplete panel"),
        )
        if verdict.accepted:
            self._on_accepted(problem, pr)

    # ---- one sample: generate (or refine), record, dedup, verify ---------
    async def _seed_pipeline(
        self,
        problem,
        *,
        role,
        round_idx,
        model,
        messages,
        seed,
        request_seed,
        identity,
        prompt_id=None,
        trial_idx=None,
        dep_proof_ids=(),
    ):
        if model.id in self.stopped:
            return
        try:
            try:
                resp = await self._do(role, model, messages, request_seed, identity)
            except RequestParked as exc:
                await self._record_error(
                    problem,
                    role,
                    round_idx,
                    exc.details,
                    seed=seed,
                    generation_model_id=model.id,
                    generation_prompt_id=prompt_id,
                    generation_random_seed=request_seed,
                    aggregation_trial_idx=trial_idx,
                )
                return
            proof, self_eval, valid = parse_proof(resp.generation, resp.finish_reason)
            await self._record_gen(
                problem,
                resp,
                role=role,
                round_idx=round_idx,
                seed=seed,
                model=model,
                request_seed=request_seed,
                prompt_id=prompt_id,
                trial_idx=trial_idx,
                dep_proof_ids=dep_proof_ids,
                proof=proof,
                self_eval=self_eval,
                valid=valid,
            )
            if model.id in self.stopped or not valid:
                return
            origin = {
                "role": role,
                "round_idx": round_idx,
                "seed": seed,
                "generation_random_seed": request_seed,
                "generation_model_id": model.id,
                "generation_prompt_id": prompt_id,
                "aggregation_trial_idx": trial_idx,
            }
            existing = self.proofs.get(proof)
            if existing is not None:  # identical proof text is verified once
                existing.origins.append(origin)
                for proof_id in dep_proof_ids:
                    if proof_id not in existing.dep_proof_ids:
                        existing.dep_proof_ids.append(proof_id)
                return
            pr = ProofRecord(
                proof=proof,
                proof_sha256=sha256_text(proof),
                self_eval=self_eval,
                self_eval_score=float(self_eval["self_eval_score"]),
                round_idx=round_idx,
                seed=seed,
                generation_model_id=model.id,
                generation_prompt_id=prompt_id,
                generation_random_seed=request_seed,
                aggregation_trial_idx=trial_idx,
                dep_proof_ids=list(dep_proof_ids),
                origins=[origin],
                proof_id=self._next_proof_id,
            )
            self._next_proof_id += 1
            self.proofs[proof] = pr
            await self._verify_proof(problem, pr)
        except asyncio.CancelledError:
            return  # stopped checkpoint or shutdown: everything completed so far is persisted

    # ---- rounds -------------------------------------------------------------
    async def _round1(self, problem):
        tasks = []
        seed = 0
        for prompt_id in self.prompts.generation_order:
            messages = prompting.user_message(
                prompting.render_generation(self.prompts.generation[prompt_id], problem["question"])
            )
            for model in self.cfg.checkpoints:
                for slot in range(model.generation_samples_per_prompt):
                    identity = resume.gen_identity(problem["problem_idx"], prompt_id, model.id, slot)
                    tasks.append(
                        self._spawn(
                            model.id,
                            self._seed_pipeline(
                                problem,
                                role=ROLE_GEN,
                                round_idx=1,
                                model=model,
                                messages=messages,
                                seed=seed,
                                request_seed=slot,
                                identity=identity,
                                prompt_id=prompt_id,
                            ),
                        )
                    )
                    seed += 1
        LOG.info("[%s] round 1: %d generation attempts", problem["problem_idx"], len(tasks))
        await self._await_batch(tasks)
        self._log_round_end(problem, 1)

    def _log_round_end(self, problem, round_idx):
        LOG.info(
            "[%s] round %d done: %d distinct candidates, %d with a complete panel, %d finalist(s)",
            problem["problem_idx"],
            round_idx,
            len(self.proofs),
            len(self._evidence()),
            len(self.finalists),
        )

    def _evidence(self):
        return [pr for pr in self.proofs.values() if pr.is_evidence]

    async def _refine_round(self, problem, round_idx):
        pool = [pr.pool_record() for pr in self._evidence()]
        parents = scoring.select_refinement_parents(pool, self.cfg.search.refinement_prompts_per_round)
        if not parents:
            LOG.info("[%s] round %d: no pool proof to refine", problem["problem_idx"], round_idx)
            return
        tasks = []
        seed = 0
        for trial_idx, parent in enumerate(parents):
            critiques = [
                c["rating"] for score in sorted(parent["score2ratings"]) for c in parent["score2ratings"][score]
            ]
            messages = prompting.user_message(
                prompting.render_refinement(self.prompts, problem["question"], parent["proof"], critiques)
            )
            for model in self.cfg.checkpoints:
                for slot in range(model.refinement_samples_per_prompt):
                    identity = resume.refine_identity(problem["problem_idx"], round_idx, trial_idx, model.id, slot)
                    tasks.append(
                        self._spawn(
                            model.id,
                            self._seed_pipeline(
                                problem,
                                role=ROLE_REFINE,
                                round_idx=round_idx,
                                model=model,
                                messages=messages,
                                seed=seed,
                                request_seed=slot,
                                identity=identity,
                                trial_idx=trial_idx,
                                dep_proof_ids=[parent["proof_id"]],
                            ),
                        )
                    )
                    seed += 1
        LOG.info(
            "[%s] round %d: refining %d pool proof(s), %d attempts",
            problem["problem_idx"],
            round_idx,
            len(parents),
            len(tasks),
        )
        await self._await_batch(tasks)
        self._log_round_end(problem, round_idx)

    def _finish(self, problem, rounds_run):
        evidence = self._evidence()
        if self.finalists:
            finalists = [self.finalists[c] for c in self.cfg.checkpoint_order if c in self.finalists]
            status = STATUS_ACCEPTED
        elif evidence:
            best = min(evidence, key=lambda pr: scoring.proof_rank_key(pr.pool_record()))
            finalists = [best]
            status = STATUS_FALLBACK
            LOG.info(
                "[%s] no accepted proof after %d round(s); fallback finalist is proof %d from %s (mean %.3f)",
                problem["problem_idx"],
                rounds_run,
                best.proof_id,
                best.generation_model_id,
                best.meanscore,
            )
        else:
            finalists = []
            status = STATUS_NO_CANDIDATES
            LOG.warning(
                "[%s] no candidate with a complete panel after %d round(s)", problem["problem_idx"], rounds_run
            )
        return SearchResult(
            status=status,
            rounds_run=rounds_run,
            finalists=finalists,
            num_candidates=len(self.proofs),
            num_parked=self.num_parked,
            num_prompt_too_long=self.num_prompt_too_long,
            pool=[pr.pool_record() for pr in evidence],
        )

    async def run(self, problem):
        try:
            await self._round1(problem)
            if self.finalists:
                return self._finish(problem, 1)
            for round_idx in range(2, self.cfg.search.max_rounds + 1):
                await self._refine_round(problem, round_idx)
                if self.finalists:
                    return self._finish(problem, round_idx)
            return self._finish(problem, self.cfg.search.max_rounds)
        except (asyncio.CancelledError, Exception):
            await self._cancel_all()
            raise
