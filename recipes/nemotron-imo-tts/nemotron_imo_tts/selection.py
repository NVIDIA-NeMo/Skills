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

"""Final candidate selection: every finalist is judged with the IMO-style prompt by every judge checkpoint.

Each judge contributes ``judgments_per_finalist`` slots. A slot whose response is truncated or has no parseable
0 to 7 score is re-sampled with a fresh seed, up to ``MAX_JUDGMENT_RESAMPLES`` extra attempts; a slot still
invalid afterwards is excluded from the mean. Finalists are ranked by the mean over valid slots, then shorter
proof text, then checkpoint order.
"""

import asyncio
from dataclasses import dataclass

from nemotron_imo_tts import prompting, resume, scoring
from nemotron_imo_tts.client import RequestParked, Response
from nemotron_imo_tts.text import is_complete_finish_reason, parse_judge_score

# Extra attempts with a fresh seed for a slot whose response is invalid (matches the competition harness).
MAX_JUDGMENT_RESAMPLES = 3
ROLE_JUDGE = "judge"
STAGE_JUDGE = "judge"
STAGE_JUDGE_ERRORS = "judge_errors"


@dataclass
class SlotResult:
    judge_id: str
    slot: int
    attempts: int
    score: int | None
    valid: bool
    seed: int


@dataclass
class FinalistJudgment:
    record: object
    slots: list
    mean_imo_score: float | None
    num_judgments: int
    num_valid_judgments: int
    num_resampled_judgments: int
    per_judge_means: dict

    def as_finalist_dict(self):
        return {
            "generation_model_id": self.record.generation_model_id,
            "proof": self.record.proof,
            "mean_imo_score": self.mean_imo_score,
            "accepted": self.record.accepted,
            "meanscore": self.record.meanscore,
        }


def _base_row(problem, record, judge, slot, attempt, seed):
    return {
        "problem_idx": problem["problem_idx"],
        "source_name": problem.get("source_name", "unknown"),
        "question": problem["question"],
        "role": ROLE_JUDGE,
        "round_idx": 0,
        "proof_sha256": record.proof_sha256,
        "generation_model_id": record.generation_model_id,
        "judge_id": judge.id,
        "slot": slot,
        "attempt": attempt,
        "seed": seed,
    }


async def _judge_slot(problem, record, judge, slot, prompts, request, sink, ledger):
    n = judge.judgments_per_finalist
    messages = prompting.user_message(prompting.render_judge(prompts.judge, problem["question"], record.proof))
    seed = slot
    for attempt in range(MAX_JUDGMENT_RESAMPLES + 1):
        seed = slot + attempt * n
        identity = resume.judge_identity(problem["problem_idx"], record.proof_sha256, judge.id, slot, attempt)
        row = ledger.lookup(identity)
        if row is not None:
            resp = Response(row.get("generation", ""), row.get("finish_reason"), None, None, None, 0, 0.0, {}, True)
        else:
            try:
                resp = await request(ROLE_JUDGE, judge, messages, seed, identity)
            except RequestParked as exc:
                error_row = _base_row(problem, record, judge, slot, attempt, seed)
                error_row.update(exc.details)
                await sink(STAGE_JUDGE_ERRORS, 0, error_row)
                return SlotResult(judge.id, slot, attempt + 1, None, False, seed)
        score = parse_judge_score(resp.generation) if is_complete_finish_reason(resp.finish_reason) else None
        if not resp.replayed:
            row = _base_row(problem, record, judge, slot, attempt, seed)
            row.update(
                {
                    "generation": resp.generation,
                    "reasoning_content": resp.reasoning_content,
                    "finish_reason": resp.finish_reason,
                    "usage": resp.usage,
                    "num_generated_tokens": resp.num_generated_tokens,
                    "attempts": resp.attempts,
                    "elapsed_s": resp.elapsed_s,
                    "context_budget": dict(resp.context_budget),
                    "score": score,
                    "valid": score is not None,
                }
            )
            await sink(STAGE_JUDGE, 0, row)
        if score is not None:
            return SlotResult(judge.id, slot, attempt + 1, score, True, seed)
    return SlotResult(judge.id, slot, MAX_JUDGMENT_RESAMPLES + 1, None, False, seed)


async def judge_finalists(problem, finalists, cfg, prompts, request, sink, ledger):
    """Run the full judge panel over every finalist concurrently and aggregate per finalist."""
    if not cfg.judges or prompts.judge is None:
        raise ValueError("judge_finalists requires a judge panel and the judge prompt")
    tasks = {}
    for index, record in enumerate(finalists):
        for judge in cfg.judges:
            for slot in range(judge.judgments_per_finalist):
                tasks[(index, judge.id, slot)] = asyncio.create_task(
                    _judge_slot(problem, record, judge, slot, prompts, request, sink, ledger)
                )
    try:
        await asyncio.gather(*tasks.values())
    except BaseException:
        for task in tasks.values():
            task.cancel()
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        raise

    judgments = []
    for index, record in enumerate(finalists):
        slots = [
            tasks[(index, judge.id, slot)].result()
            for judge in cfg.judges
            for slot in range(judge.judgments_per_finalist)
        ]
        valid = [s.score for s in slots if s.valid]
        per_judge_means = {}
        for judge in cfg.judges:
            scores = [s.score for s in slots if s.judge_id == judge.id and s.valid]
            per_judge_means[judge.id] = sum(scores) / len(scores) if scores else None
        judgments.append(
            FinalistJudgment(
                record=record,
                slots=slots,
                mean_imo_score=sum(valid) / len(valid) if valid else None,
                num_judgments=len(slots),
                num_valid_judgments=len(valid),
                num_resampled_judgments=sum(1 for s in slots if s.attempts > 1),
                per_judge_means=per_judge_means,
            )
        )
    return judgments


def select(judgments, cfg):
    """The finalist to submit: best mean IMO score, then shorter proof, then checkpoint order."""
    return min(
        judgments, key=lambda j: scoring.finalist_rank_key(j.as_finalist_dict(), cfg.checkpoint_order, judged=True)
    )


def select_unjudged(finalists, cfg):
    """Without a judge panel: accepted first, then mean verifier score, shorter proof, checkpoint order."""
    return min(
        finalists,
        key=lambda r: scoring.finalist_rank_key(
            {
                "generation_model_id": r.generation_model_id,
                "proof": r.proof,
                "accepted": r.accepted,
                "meanscore": r.meanscore,
            },
            cfg.checkpoint_order,
            judged=False,
        ),
    )
