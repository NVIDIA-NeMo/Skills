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

"""The rules the report fixes: acceptance, proof ranking, critique retention, and finalist ranking.

Pure functions over plain dicts so the live pipeline and offline replays share one definition.
"""

import collections
from dataclasses import dataclass

# A refinement prompt carries at most this many verifier critiques, and at most this many per score value
# when the retained critiques have mixed scores.
MAX_CRITIQUES = 8
MAX_CRITIQUES_PER_SCORE = 4


@dataclass(frozen=True)
class PanelVerdict:
    complete: bool
    accepted: bool
    meanscore: float
    verifier_means: dict
    valid_scores: list


def panel_verdict(scores_by_verifier, judgments_per_proof, interrupted=False):
    """Judge a proof from its verifier panel.

    ``scores_by_verifier`` maps each verifier id to the scores of the responses received (``None`` for an
    invalid judgment). The panel is complete only when every verifier returned exactly ``judgments_per_proof``
    valid judgments and nothing was interrupted. Accepted means complete and every judgment equal to 1.
    The mean score is the mean of the per-verifier means, so verifiers weigh equally.
    """
    valid_by_verifier = {v: [s for s in scores if s is not None] for v, scores in scores_by_verifier.items()}
    complete = not interrupted and all(len(scores) == judgments_per_proof for scores in valid_by_verifier.values())
    if not complete:
        return PanelVerdict(False, False, 0.0, {v: 0.0 for v in scores_by_verifier}, [])
    verifier_means = {v: sum(scores) / len(scores) for v, scores in valid_by_verifier.items()}
    meanscore = sum(verifier_means.values()) / len(verifier_means)
    valid_scores = [s for v in scores_by_verifier for s in valid_by_verifier[v]]
    return PanelVerdict(True, all(s == 1.0 for s in valid_scores), meanscore, verifier_means, valid_scores)


def proof_rank_key(record):
    """Sort key for pool proofs: best first by mean verifier score, then self-evaluation, later round, proof id."""
    return (
        -float(record.get("meanscore", 0.0) or 0.0),
        -float(record.get("self_eval_score", 0.0) or 0.0),
        -int(record.get("round_idx", 0) or 0),
        int(record.get("proof_id", 0) or 0),
    )


def select_refinement_parents(records, k):
    return sorted(records, key=proof_rank_key)[:k]


def retain_critiques(candidates, verifier_order):
    """Keep at most 8 critiques, balanced across verifiers, at most 4 per score value when scores are mixed.

    Candidates are dicts with ``rating``, ``score``, ``verifier_id``, ``verification_seed``. Selection walks the
    verifiers in order, taking their next eligible judgment by seed, and stops as soon as any verifier runs out.
    The result is ordered by (score, verifier order, seed) so refinement prompts are deterministic.
    """
    if not candidates or not verifier_order:
        return []
    per_verifier_limit = MAX_CRITIQUES // len(verifier_order)
    scores = {c["score"] for c in candidates}
    per_score_limit = MAX_CRITIQUES if len(scores) == 1 else MAX_CRITIQUES_PER_SCORE
    queues = {
        v: sorted((c for c in candidates if c["verifier_id"] == v), key=lambda c: c["verification_seed"])
        for v in verifier_order
    }
    positions = {v: 0 for v in verifier_order}
    score_counts = collections.defaultdict(int)
    selected = []
    for _ in range(per_verifier_limit):
        batch = []
        batch_counts = collections.defaultdict(int)
        next_positions = {}
        for v in verifier_order:
            queue, position, pick = queues[v], positions[v], None
            while position < len(queue):
                item = queue[position]
                position += 1
                if score_counts[item["score"]] + batch_counts[item["score"]] >= per_score_limit:
                    continue
                pick = item
                break
            if pick is None:
                return _ordered(selected, verifier_order)
            batch.append(pick)
            batch_counts[pick["score"]] += 1
            next_positions[v] = position
        selected.extend(batch)
        positions.update(next_positions)
        for score, count in batch_counts.items():
            score_counts[score] += count
    return _ordered(selected, verifier_order)


def _ordered(selected, verifier_order):
    rank = {v: i for i, v in enumerate(verifier_order)}
    return sorted(selected, key=lambda c: (c["score"], rank[c["verifier_id"]], c["verification_seed"]))


def finalist_rank_key(finalist, checkpoint_order, judged):
    """Sort key for finalists: best first.

    With a judge panel: higher mean IMO score, then shorter proof, then checkpoint order; a finalist without any
    valid judgment ranks last. Without judges: accepted first, then mean verifier score, shorter proof, order.
    """
    order = checkpoint_order.index(finalist["generation_model_id"])
    if judged:
        score = finalist.get("mean_imo_score")
        return (score is None, -(score or 0.0), len(finalist["proof"]), order)
    return (
        not finalist.get("accepted", False),
        -float(finalist.get("meanscore", 0.0) or 0.0),
        len(finalist["proof"]),
        order,
    )
