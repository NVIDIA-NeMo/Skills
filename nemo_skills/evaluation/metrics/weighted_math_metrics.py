# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import math
from collections import Counter, defaultdict

import numpy as np

from nemo_skills.evaluation.metrics.base import as_percentage
from nemo_skills.evaluation.metrics.math_metrics import MathMetrics


class WeightedMathMetrics(MathMetrics):
    """MathMetrics with difficulty-weighted accuracy."""

    def reset(self) -> None:
        super().reset()
        self.total_weight = 0.0
        self.weighted_sums = defaultdict(lambda: defaultdict(float))
        self.per_sample_scores = defaultdict(list)

    def _get_sample_weight(self, prediction: dict) -> float:
        """Return the sample weight, defaulting to 1.0 if the field is absent."""
        return float(prediction.get("weight", 1.0))

    def _update_pass1_avg_of_k(self, score_method: str, attempt_scores: list[bool], sample_weight: float) -> None:
        """Accumulate weighted average correctness across first k attempts for all k."""
        num_attempts = len(attempt_scores)
        for k in range(1, num_attempts + 1):
            self.weighted_sums[f"pass@1[avg-of-{k}]"][score_method] += sample_weight * sum(attempt_scores[:k]) / k

    def _update_pass_at_k(self, score_method: str, attempt_scores: list[bool], sample_weight: float) -> None:
        """Accumulate weighted combinatorial pass-at-k probability for all k."""
        num_attempts = len(attempt_scores)
        num_incorrect = attempt_scores.count(False)
        for k in range(1, num_attempts + 1):
            if num_incorrect < k:
                pass_k_score = 1.0
            else:
                pass_k_score = 1.0 - math.comb(num_incorrect, k) / math.comb(num_attempts, k)
            self.weighted_sums[f"pass@{k}"][score_method] += sample_weight * pass_k_score

    def _update_majority_at_k(
        self, score_method: str, attempt_scores: list[bool], predicted_answers: list, sample_weight: float
    ) -> None:
        """Accumulate weighted majority-vote accuracy for all k >= 2."""
        num_attempts = len(attempt_scores)
        for k in range(2, num_attempts + 1):
            valid_pairs = [(ans, sc) for ans, sc in zip(predicted_answers[:k], attempt_scores[:k]) if ans is not None]
            if not valid_pairs:
                majority_score = 0.0
            else:
                answer_counts = Counter(valid_pairs)
                top_count = answer_counts.most_common(1)[0][1]
                top_pairs = [(ans, sc) for (ans, sc), cnt in answer_counts.items() if cnt == top_count]
                majority_score = sum(sc for _, sc in top_pairs) / len(top_pairs)
            self.weighted_sums[f"majority@{k}"][score_method] += sample_weight * majority_score

    def update(self, predictions: list[dict]) -> None:
        super().update(predictions)

        sample_weight = self._get_sample_weight(predictions[0])
        self.total_weight += sample_weight

        score_dicts = [self._get_score_dict(p) for p in predictions]
        predicted_answers = [p[self.answer_key] for p in predictions]

        all_score_methods = set().union(*[sd.keys() for sd in score_dicts])

        for score_method in all_score_methods:
            attempt_scores = [bool(sd.get(score_method, False)) for sd in score_dicts]
            self.per_sample_scores[score_method].append((sample_weight, attempt_scores))

            self._update_pass1_avg_of_k(score_method, attempt_scores, sample_weight)
            self._update_pass_at_k(score_method, attempt_scores, sample_weight)
            self._update_majority_at_k(score_method, attempt_scores, predicted_answers, sample_weight)

    def _add_weighted_std_metrics(self, metrics_dict: dict) -> None:
        """Populate weighted_<scoring_method>_statistics for pass@1[avg-of-k] with k >= 2."""
        if self.max_k < 2 or not self.per_sample_scores:
            return
        for score_method, sample_entries in self.per_sample_scores.items():
            for k in range(2, self.max_k + 1):
                agg_key = f"pass@1[avg-of-{k}]"
                if agg_key not in metrics_dict:
                    continue
                # Weighted average correctness at each attempt position (across all samples)
                weighted_avg_per_attempt = [
                    sum(weight * scores[attempt_idx] for weight, scores in sample_entries) / self.total_weight
                    for attempt_idx in range(k)
                ]
                std_dev = np.std(weighted_avg_per_attempt, ddof=1)
                std_err = std_dev / math.sqrt(k)
                avg_sample_std = (
                    sum(weight * np.std(scores[:k], ddof=1) for weight, scores in sample_entries) / self.total_weight
                )
                avg = np.mean(weighted_avg_per_attempt)
                metrics_dict[agg_key][f"weighted_{score_method}_statistics"] = {
                    "avg": float(avg),
                    "std_dev_across_runs": float(std_dev),
                    "std_err_across_runs": float(std_err),
                    "avg_sample_std_dev": float(avg_sample_std),
                }

    def get_metrics(self) -> dict:
        metrics = super().get_metrics()
        if self.total_weight > 0:
            for agg_mode, score_method_sums in self.weighted_sums.items():
                for score_method, weighted_sum in score_method_sums.items():
                    weighted_acc = 100.0 * weighted_sum / self.total_weight
                    if agg_mode in metrics:
                        metrics[agg_mode][f"weighted_{score_method}"] = weighted_acc
        self._add_weighted_std_metrics(metrics)
        return metrics

    def metrics_to_print(self) -> dict:
        result = super().metrics_to_print()
        result["weighted_symbolic_correct"] = as_percentage
        result["weighted_judge_correct"] = as_percentage
        result["weighted_both_correct"] = as_percentage
        result["weighted_any_correct"] = as_percentage
        return result
