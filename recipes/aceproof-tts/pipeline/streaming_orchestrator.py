"""Per-problem streaming orchestrator for the aceproof-tts pipeline.

This is the core async state machine that replaces the blocking SLURM DAG
(prepare -> proof_gen -> aggregate -> verify -> prepare_refine -> refine ->
aggregate -> verify -> finalize) with a single per-problem loop that advances
each sample gen -> verify -> score independently and streams a result as soon
as the problem is done.

It is framework-agnostic: it receives an async `request(role, messages, seed)`
callable (wired by the GenerationTask subclass to skills' bounded model client)
and a `sink` for writing schema-compatible round records. All request
concurrency is owned by the caller's semaphore + transport; this module only
owns scheduling, early-stop, solved-and-stop and refinement logic.

Reuses the existing pipeline helpers (parsing, pool, refinement) so the on-disk
outputs stay compatible with finalize_results.py / metrics tooling.
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from prepare_refinement import _build_refinement_tasks
from proof_pool_manager import ProofPoolManager
from utils import (
    extract_boxed_answers,
    extract_self_eval,
    extract_solution,
    is_complete_finish_reason,
    parse_verification_score,
    response_metadata,
    strip_think,
)

# role -> stage name used in the on-disk round directory layout.
ROLE_GEN = "gen"
ROLE_VERIFY = "verify"
ROLE_REFINE = "refine"


def parse_proof(generation, finish_reason):
    """Mirror of scripts/proof_generation.process_single proof extraction."""
    complete = is_complete_finish_reason(finish_reason)
    if not complete:
        return "UNFINISHED PROOF GENERATION", {"self_eval": "null", "self_eval_score": 0}, False
    text = strip_think(generation)
    try:
        self_eval_text = extract_self_eval(text).strip()
        solution_text = extract_solution(text).strip()
    except Exception:
        return text.strip(), {"self_eval": "null", "self_eval_score": 0}, bool(text.strip())
    score = 0.0
    try:
        scores = [s.strip() for s in extract_boxed_answers(self_eval_text) if s.strip()]
        if scores:
            score = float(scores[-1])
    except Exception:
        score = 0.0
    self_eval = {"self_eval": self_eval_text, "self_eval_score": score}
    return solution_text, self_eval, bool(solution_text)


@dataclass
class StreamingConfig:
    n_parallel_proof_gen: int
    n_verification_per_proof: int
    n_agg_trials: int
    n_best_proofs_to_sample: int
    n_proofs_to_refine: int
    max_rating_per_score: int
    n_samples_per_trial: int
    solved_threshold: float
    max_rounds: int
    # early-stop experiment knobs
    min_verifications_per_proof: int
    early_stop_only_if_score_lt_1: bool
    cancel_remaining: bool


@dataclass
class ProofRecord:
    proof: str
    self_eval: dict
    round_idx: int
    seed: int
    dep_proof_ids: list = field(default_factory=list)
    ratings: list = field(default_factory=list)  # list of {"rating", "score"}
    valid_scores: list = field(default_factory=list)
    meanscore: float = 0.0
    broke_early: bool = False
    verify_tasks: list = field(default_factory=list)


# request signature: async (role, messages, seed) -> response dict
RequestFn = Callable[[str, list, int], Awaitable[dict]]
# sink signature: async (stage, round_idx, row) -> None
SinkFn = Callable[[str, int, dict], Awaitable[None]]


class ProblemOrchestrator:
    """Runs the full gen -> verify -> score -> (solved? / refine) loop for ONE problem."""

    def __init__(
        self,
        cfg: StreamingConfig,
        request: RequestFn,
        sink: SinkFn,
        gen_prompt_template: str,
        verify_prompt_template: str,
        refine_prompt_template: str,
        proof_gen_prompt_template: str,
        gen_system_prompt: Optional[str] = None,
        verify_system_prompt: Optional[str] = None,
        refine_system_prompt: Optional[str] = None,
        pool_manager: Optional[ProofPoolManager] = None,
    ):
        self.cfg = cfg
        self.request = request
        self.sink = sink
        self.gen_prompt_template = gen_prompt_template
        self.verify_prompt_template = verify_prompt_template
        self.refine_prompt_template = refine_prompt_template
        self.proof_gen_prompt_template = proof_gen_prompt_template
        self.gen_system_prompt = gen_system_prompt
        self.verify_system_prompt = verify_system_prompt
        self.refine_system_prompt = refine_system_prompt
        self.pool_manager = pool_manager

        self.proofs: dict[str, ProofRecord] = {}
        self.solved = False
        self.solved_record: Optional[ProofRecord] = None
        self._all_tasks: set[asyncio.Task] = set()

    # ---- message builders -------------------------------------------------
    def _messages(self, system_prompt, user_text):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})
        return messages

    def _gen_messages(self, problem):
        return self._messages(self.gen_system_prompt, self.gen_prompt_template.format(question=problem["question"]))

    def _verify_messages(self, problem, proof):
        user = self.verify_prompt_template.format(statement=problem["question"], proof=proof)
        return self._messages(self.verify_system_prompt, user)

    # ---- task bookkeeping / cancellation ---------------------------------
    def _register(self, tasks):
        for t in tasks:
            self._all_tasks.add(t)

    async def _cancel(self, tasks):
        pending = [t for t in tasks if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    def _declare_solved(self, record):
        # set the latch BEFORE cancelling so no completion handler starts new work
        self.solved = True
        self.solved_record = record
        if self.cfg.cancel_remaining:
            for t in list(self._all_tasks):
                if not t.done():
                    t.cancel()

    # ---- request wrapper --------------------------------------------------
    async def _do(self, role, messages, seed):
        return await self.request(role, messages, seed)

    # ---- record builders (schema-compatible with the batch scripts) ------
    def _base_row(self, problem):
        row = dict(problem)
        for key in ("messages", "prompt", "system_prompt", "_async_position"):
            row.pop(key, None)
        return row

    async def _record_gen(self, problem, seed, round_idx, resp, proof, self_eval, valid, role):
        generation = resp.get("generation", "")
        finish_reason = resp.get("finish_reason")
        row = self._base_row(problem)
        row.update(response_metadata(resp, "proof_generation"))
        row.update(
            {
                "generation_seed": seed,
                "row_id": f"{problem['problem_idx']}_{seed}",
                "generation": generation,
                "proof": proof,
                "self_eval": self_eval,
                "self_eval_score": self_eval.get("self_eval_score", 0),
                "finish_reason": finish_reason,
                "generation_complete": is_complete_finish_reason(finish_reason),
                "valid": valid,
            }
        )
        stage = ROLE_REFINE if role == ROLE_REFINE else "proof_gen"
        await self.sink(stage, round_idx, row)

    async def _record_verify(self, problem, pr, vseed, resp):
        generation = resp.get("generation", "")
        finish_reason = resp.get("finish_reason")
        rating_text = strip_think(generation)
        complete = is_complete_finish_reason(finish_reason)
        score = parse_verification_score(rating_text) if complete else None
        row = self._base_row(problem)
        row.update(response_metadata(resp, "verification"))
        row.update(
            {
                "proof": pr.proof,
                "self_eval": pr.self_eval,
                "dep_proof_ids": pr.dep_proof_ids,
                "generation_seed": pr.seed,
                "row_id": f"{problem['problem_idx']}_{pr.seed}",
                "verification_seed": vseed,
                "verify_row_id": f"{problem['problem_idx']}_{pr.seed}_v{vseed}",
                "generation": generation,
                "rating_text": rating_text,
                "verification_score": score,
                "finish_reason": finish_reason,
                "verification_complete": complete,
                "valid": score is not None and complete,
            }
        )
        await self.sink(ROLE_VERIFY, pr.round_idx, row)
        return rating_text, score

    # ---- core loops -------------------------------------------------------
    async def _verify_proof(self, problem, pr):
        msgs = self._verify_messages(problem, pr.proof)
        tasks = [
            asyncio.create_task(self._do(ROLE_VERIFY, msgs, vseed))
            for vseed in range(self.cfg.n_verification_per_proof)
        ]
        pr.verify_tasks = tasks
        self._register(tasks)

        valid_scores = []
        broke_early = False
        pending = set(tasks)
        try:
            while pending and not self.solved and not broke_early:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for d in done:
                    if d.cancelled():
                        continue
                    try:
                        resp = d.result()
                    except Exception:
                        continue
                    vseed = len(pr.ratings)
                    rating_text, score = await self._record_verify(problem, pr, vseed, resp)
                    if score is None:
                        continue
                    pr.ratings.append({"rating": rating_text, "score": score})
                    valid_scores.append(score)
                    if len(valid_scores) >= self.cfg.min_verifications_per_proof:
                        mean = sum(valid_scores) / len(valid_scores)
                        should_stop = (mean < 1.0) or (not self.cfg.early_stop_only_if_score_lt_1)
                        if should_stop:
                            broke_early = True
                            break
        finally:
            if self.cfg.cancel_remaining:
                await self._cancel(tasks)

        pr.valid_scores = valid_scores
        pr.meanscore = (sum(valid_scores) / len(valid_scores)) if valid_scores else 0.0
        pr.broke_early = broke_early

        # solved-and-stop: only a proof that consumed its FULL budget (no early
        # cancel) with meanscore over threshold counts as fully correct.
        if (not broke_early) and valid_scores and pr.meanscore > self.cfg.solved_threshold and not self.solved:
            self._declare_solved(pr)

    async def _seed_pipeline(self, problem, messages, seed, round_idx, role, dep_proof_ids=None):
        if self.solved:
            return
        try:
            resp = await self._do(role, messages, seed)
        except asyncio.CancelledError:
            return
        except Exception:
            return
        if self.solved:  # race guard: do not fan out if solved while we generated
            return
        proof, self_eval, valid = parse_proof(resp.get("generation", ""), resp.get("finish_reason"))
        await self._record_gen(problem, seed, round_idx, resp, proof, self_eval, valid, role)
        if not valid:
            return
        if proof in self.proofs:  # dedup: a proof string is verified once
            return
        pr = ProofRecord(
            proof=proof,
            self_eval=self_eval,
            round_idx=round_idx,
            seed=seed,
            dep_proof_ids=list(dep_proof_ids or []),
        )
        self.proofs[proof] = pr
        await self._verify_proof(problem, pr)

    async def _refine_round(self, problem, round_idx):
        pool_records = self._pool_records_for_refine()
        if not pool_records:
            return
        base_item = self._base_row(problem)
        tasks = _build_refinement_tasks(
            base_item=base_item,
            proof_records=pool_records,
            num_trials=self.cfg.n_agg_trials,
            n_best_proofs_to_sample=self.cfg.n_best_proofs_to_sample,
            n_proofs_to_refine=self.cfg.n_proofs_to_refine,
            max_rating_per_score=self.cfg.max_rating_per_score,
            refine_prompt_template=self.refine_prompt_template,
            proof_gen_prompt_template=self.proof_gen_prompt_template,
            system_prompt=self.refine_system_prompt,
        )
        if not tasks:
            return

        seed_tasks = []
        seed_counter = 0
        for trial in tasks:
            messages = trial.get("messages") or self._messages(self.refine_system_prompt, trial.get("prompt", ""))
            dep_proof_ids = trial.get("dep_proof_ids", [])
            for _ in range(max(1, self.cfg.n_samples_per_trial)):
                seed = seed_counter
                seed_counter += 1
                seed_tasks.append(
                    asyncio.create_task(
                        self._seed_pipeline(problem, messages, seed, round_idx, ROLE_REFINE, dep_proof_ids)
                    )
                )
        self._register(seed_tasks)
        await asyncio.gather(*seed_tasks, return_exceptions=True)

    # ---- pool / result helpers -------------------------------------------
    def _pool_records_for_refine(self):
        records = []
        proof_id = 1
        for proof, pr in self.proofs.items():
            if not pr.valid_scores:
                continue
            score2ratings = defaultdict(list)
            for r in pr.ratings:
                score2ratings[r["score"]].append(r)
            records.append(
                {
                    "proof": proof,
                    "meanscore": pr.meanscore,
                    "score2ratings": dict(score2ratings),
                    "self_eval": pr.self_eval,
                    "dep_proof_ids": pr.dep_proof_ids,
                    "proof_id": proof_id,
                }
            )
            proof_id += 1
        return records

    def _best_proof(self):
        scored = [pr for pr in self.proofs.values() if pr.valid_scores]
        if not scored:
            return None

        def sort_key(pr):
            se = pr.self_eval.get("self_eval_score", 0) if isinstance(pr.self_eval, dict) else 0
            return (pr.meanscore, se, pr.round_idx)

        return max(scored, key=sort_key)

    def _ingest_pool(self, problem):
        """Write the per-problem proof pool on disk so finalize_results.py stays usable."""
        if self.pool_manager is None:
            return
        source_name = problem.get("source_name", "unknown")
        problem_idx = problem["problem_idx"]
        by_round = defaultdict(list)
        for pr in self.proofs.values():
            if not pr.valid_scores:
                continue
            score2ratings = defaultdict(list)
            for r in pr.ratings:
                score2ratings[r["score"]].append(r)
            by_round[pr.round_idx].append(
                {
                    "proof": pr.proof,
                    "meanscore": pr.meanscore,
                    "score2ratings": dict(score2ratings),
                    "self_eval": pr.self_eval,
                    "dep_proof_ids": pr.dep_proof_ids,
                }
            )
        for round_idx in sorted(by_round):
            self.pool_manager.ingest_new_records(source_name, problem_idx, by_round[round_idx], round_idx)

    def _emit(self, problem, record, solved, round_idx):
        self._ingest_pool(problem)
        out = self._base_row(problem)
        if record is None:
            out.update(
                {
                    "generation": "",
                    "proof": "",
                    "meanscore": 0.0,
                    "self_eval": {"self_eval": "null", "self_eval_score": 0},
                    "solved": False,
                    "result_round_idx": round_idx,
                    "num_proofs": len(self.proofs),
                }
            )
            return out
        out.update(
            {
                "generation": record.proof,
                "proof": record.proof,
                "meanscore": record.meanscore,
                "self_eval": record.self_eval,
                "self_eval_score": record.self_eval.get("self_eval_score", 0)
                if isinstance(record.self_eval, dict)
                else 0,
                "solved": bool(solved),
                "result_round_idx": record.round_idx,
                "num_proofs": len(self.proofs),
                "num_verifications": len(record.valid_scores),
                "early_stopped": record.broke_early,
            }
        )
        return out

    async def run(self, problem):
        # ---- Round 1: independent per-seed gen -> verify -> score ----
        gen_msgs = self._gen_messages(problem)
        seed_tasks = [
            asyncio.create_task(self._seed_pipeline(problem, gen_msgs, seed, 1, ROLE_GEN))
            for seed in range(self.cfg.n_parallel_proof_gen)
        ]
        self._register(seed_tasks)
        await asyncio.gather(*seed_tasks, return_exceptions=True)

        if self.solved:
            return self._emit(problem, self.solved_record, solved=True, round_idx=1)

        # ---- Round 2+: per-problem refinement ----
        for round_idx in range(2, self.cfg.max_rounds + 1):
            if self.solved:
                break
            await self._refine_round(problem, round_idx)
            if self.solved:
                return self._emit(problem, self.solved_record, solved=True, round_idx=round_idx)

        return self._emit(problem, self._best_proof(), solved=False, round_idx=self.cfg.max_rounds)
