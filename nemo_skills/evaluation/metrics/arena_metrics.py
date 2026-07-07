# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import re
from collections import defaultdict
from statistics import mean

from nemo_skills.evaluation.metrics.base import BaseMetrics

# Score-label preference for picking the best of N predictions per judgement direction.
# "Best" = most candidate-favorable. The two preference orders are mirrored because the
# judge prompts swap the A/B slot assignments to mitigate position bias:
#   judgement-gen-base: A = candidate's answer, B = baseline's answer
#   judgement-base-gen: A = baseline's answer, B = candidate's answer
_GEN_BASE_PREFERENCE = ("A>>B", "A>B", "A=B", "B>A", "B>>A")
_BASE_GEN_PREFERENCE = ("B>>A", "B>A", "A=B", "A>B", "A>>B")


class ArenaMetrics(BaseMetrics):
    def __init__(self):
        self.reset()

    def _get_judge_score(self, judgment):
        # adapted from https://github.com/lm-sys/arena-hard-auto/blob/main/gen_judgment.py
        pattern = re.compile("\[\[([AB<>=]+)\]\]")
        matches = pattern.findall(judgment)
        matches = [m for m in matches if m != ""]
        if len(set(matches)) == 0:
            return None
        elif len(set(matches)) == 1:
            return matches[0].strip("\n")
        else:
            return None

    def get_incorrect_sample(self, prediction: dict) -> dict:
        prediction = prediction.copy()
        prediction["judgement-gen-base"] = "Rating: [[A>>B]]"
        prediction["judgement-base-gen"] = "Rating: [[B>>A]]"
        return prediction

    @staticmethod
    def _best_pair(prompt_pairs):
        """Pick the most candidate-favorable label across all predictions, per direction."""
        gen_base_pool = [pair[0] for pair in prompt_pairs]
        base_gen_pool = [pair[1] for pair in prompt_pairs]
        return [
            next((s for s in _GEN_BASE_PREFERENCE if s in gen_base_pool), None),
            next((s for s in _BASE_GEN_PREFERENCE if s in base_gen_pool), None),
        ]

    def update(self, predictions):
        """Store all per-prediction (gen-base, base-gen) score pairs for this prompt.

        Aggregation is deferred to get_metrics() so that both pass@N (best-of-N) and
        pass@1[avg-of-N] can be derived from the same stored data.
        """
        super().update(predictions)
        self.per_prompt_scores.append(
            [
                (
                    self._get_judge_score(p["judgement-gen-base"]),
                    self._get_judge_score(p["judgement-base-gen"]),
                )
                for p in predictions
            ]
        )
        self.categories.append(predictions[0].get("category"))

    def get_metrics(self):
        n = self.max_k or 1
        emit_categories = len(set(self.categories)) > 1

        # pass@N (best-of-N): pick the most candidate-favorable label per direction across
        # all N predictions per prompt, then run Elo on those 1-pair-per-prompt lists.
        best_of_n = [self._best_pair(pairs) for pairs in self.per_prompt_scores]
        metrics_dict = {f"pass@{n}": self._aggregate(best_of_n, emit_categories)}

        # pass@1[avg-of-N]: N independent single-shot Elo bootstraps (one per repeat),
        # averaged. Skipped for N==1 since avg-of-1 is degenerate with pass@1.
        if n > 1:
            per_repeat_aggs = [
                self._aggregate(
                    [list(pairs[r]) for pairs in self.per_prompt_scores],
                    emit_categories,
                )
                for r in range(n)
            ]
            metrics_dict[f"pass@1[avg-of-{n}]"] = self._average_aggregations(per_repeat_aggs)

        return metrics_dict

    def _aggregate(self, prompt_pairs, emit_categories):
        """Run get_aggregate_score on a list of (gen-base, base-gen) pairs (one per prompt)."""
        from nemo_skills.evaluation.evaluator.arena import get_aggregate_score

        agg = {"num_entries": self.total}
        agg.update(self._native_aggregate_score(get_aggregate_score(prompt_pairs)))
        self.update_common_metrics(agg)

        if emit_categories:
            by_category = defaultdict(list)
            for pair, category in zip(prompt_pairs, self.categories, strict=True):
                by_category[category].append(pair)
            for category, pairs in by_category.items():
                cat_agg = {"num_entries": len(pairs)}
                cat_agg.update(self._native_aggregate_score(get_aggregate_score(pairs)))
                agg[f"category_{category}"] = cat_agg

        return agg

    @staticmethod
    def _native_aggregate_score(agg):
        """Cast get_aggregate_score's numpy types to native Python — yaml.safe_dump can't serialize numpy."""
        return {
            "score": float(agg["score"]),
            "95_CI": tuple(float(x) for x in agg["95_CI"]),
            "invalid_scores": int(agg["invalid_scores"]),
        }

    def _average_aggregations(self, per_repeat):
        """Average a list of per-repeat aggregation dicts to produce pass@1[avg-of-N].

        - 'score': mean across repeats.
        - 'invalid_scores': summed across repeats (total invalid-judgement count).
        - '95_CI': dropped (mean of CIs is not a meaningful CI).
        - num_entries / avg_tokens / gen_seconds: same across repeats; populated by
          update_common_metrics.
        - Per-category sub-dicts: averaged using the same rules.

        Per-repeat scores are not surfaced here because downstream metric parsers
        (e.g. nemo-evaluator-launcher's `core_evals/nemo_skills/output.py`) wrap each
        leaf value in a `Score(value=float)` and reject lists. Consumers who need the
        per-repeat breakdown can recompute it from `output-rs*.jsonl`.
        """
        # Cast to native Python float — get_aggregate_score returns numpy.float64,
        # which yaml.safe_dump can't serialize.
        avg = {"num_entries": per_repeat[0]["num_entries"]}
        avg["score"] = float(mean(m["score"] for m in per_repeat))
        avg["invalid_scores"] = sum(m["invalid_scores"] for m in per_repeat)
        self.update_common_metrics(avg)

        for cat_key in [k for k in per_repeat[0] if k.startswith("category_")]:
            cat_avg = {"num_entries": per_repeat[0][cat_key]["num_entries"]}
            cat_avg["score"] = float(mean(m[cat_key]["score"] for m in per_repeat))
            cat_avg["invalid_scores"] = sum(m[cat_key]["invalid_scores"] for m in per_repeat)
            avg[cat_key] = cat_avg

        return avg

    def evaluations_to_print(self):
        # Override BaseMetrics' default — Arena doesn't compute majority@k, so dropping
        # that key avoids a missing-key request to the framework's printer (matches the
        # OmniMetrics convention).
        n = self.max_k or 1
        if n > 1:
            return [f"pass@{n}", f"pass@1[avg-of-{n}]"]
        return ["pass@1"]

    def reset(self):
        super().reset()
        # Per-prompt list of (gen-base, base-gen) score pairs — N tuples per prompt where
        # N == self.max_k. Aggregation is deferred to get_metrics() so both pass@N
        # (best-of-N) and pass@1[avg-of-N] can be derived from the same data.
        self.per_prompt_scores = []
        self.categories = []
