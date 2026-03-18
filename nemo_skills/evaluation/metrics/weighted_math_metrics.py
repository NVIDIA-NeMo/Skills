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
    """MathMetrics with difficulty-weighted accuracy.

    Reads the 'weight' field written by prepare.py.
    Weighted accuracy = sum(weight_i * correct_i) / sum(weight_i).
    Falls back to weight=1 if field is missing.

    Computes weighted_<signal> for all signals present in score_dicts
    (symbolic_correct, judge_correct, both_correct, any_correct) across
    all eval modes: pass@1[avg-of-k], pass@k, majority@k (k >= 2).
    Computes weighted_<signal>_statistics for pass@1[avg-of-k].
    """

    def reset(self):
        super().reset()
        self.total_weight = 0.0
        # weighted_eval_dict[agg_mode][signal] -> weighted sum
        self.weighted_eval_dict = defaultdict(lambda: defaultdict(float))
        # all_weighted_scores[signal] -> list of (weight, scores_list)
        self.all_weighted_scores: dict[str, list[tuple[float, list[bool]]]] = defaultdict(list)

    def update(self, predictions: list[dict]):
        # Parent handles pass@k, majority@k, reward@k for symbolic_correct
        super().update(predictions)

        weight = float(predictions[0].get("weight", 1.0))
        self.total_weight += weight

        score_dicts = [self._get_score_dict(p) for p in predictions]
        predicted_answers = [p[self.answer_key] for p in predictions]

        all_signal_names = set().union(*[sd.keys() for sd in score_dicts])

        for signal in all_signal_names:
            scores = [bool(sd.get(signal, False)) for sd in score_dicts]
            self.all_weighted_scores[signal].append((weight, scores))

            total = len(scores)
            total_incorrect = scores.count(False)

            for k in range(1, total + 1):
                # pass@1[avg-of-k]: average correctness across first k attempts
                self.weighted_eval_dict[f"pass@1[avg-of-{k}]"][signal] += weight * sum(scores[:k]) / k

                # pass@k (binary combinatorial formula)
                if total_incorrect < k:
                    pass_k_score = 1.0
                else:
                    pass_k_score = 1.0 - math.comb(total_incorrect, k) / math.comb(total, k)
                self.weighted_eval_dict[f"pass@{k}"][signal] += weight * pass_k_score

            # majority@k (k >= 2)
            for k in range(2, total + 1):
                valid = [(ans, sc) for ans, sc in zip(predicted_answers[:k], scores[:k]) if ans is not None]
                if not valid:
                    majority_score = 0.0
                else:
                    counter = Counter(valid)
                    majority_count = counter.most_common(1)[0][1]
                    majority_answer_list = [(a, s) for (a, s), cnt in counter.items() if cnt == majority_count]
                    majority_score = sum(s for _, s in majority_answer_list) / len(majority_answer_list)
                self.weighted_eval_dict[f"majority@{k}"][signal] += weight * majority_score

    def _add_weighted_std_metrics(self, metrics_dict):
        if self.max_k < 2 or not self.all_weighted_scores:
            return
        for signal, weighted_scores_list in self.all_weighted_scores.items():
            for k in range(2, self.max_k + 1):
                key = f"pass@1[avg-of-{k}]"
                if key not in metrics_dict:
                    continue
                run_means = [
                    sum(w * scores[j] for w, scores in weighted_scores_list) / self.total_weight for j in range(k)
                ]
                std_dev = np.std(run_means, ddof=1)
                std_err = std_dev / math.sqrt(k)
                avg_sample_std = float(
                    sum(w * np.std(scores[:k], ddof=1) for w, scores in weighted_scores_list) / self.total_weight
                )
                avg = float(np.mean(run_means))
                metrics_dict[key][f"weighted_{signal}_statistics"] = {
                    "avg": avg,
                    "std_dev_across_runs": float(std_dev),
                    "std_err_across_runs": float(std_err),
                    "avg_sample_std_dev": avg_sample_std,
                }

    def get_metrics(self) -> dict:
        metrics = super().get_metrics()
        if self.total_weight > 0:
            for agg_mode, signal_sums in self.weighted_eval_dict.items():
                for signal, weighted_sum in signal_sums.items():
                    weighted_acc = 100.0 * weighted_sum / self.total_weight
                    if agg_mode in metrics:
                        metrics[agg_mode][f"weighted_{signal}"] = weighted_acc
        self._add_weighted_std_metrics(metrics)
        return metrics

    def metrics_to_print(self):
        result = super().metrics_to_print()
        result["weighted_symbolic_correct"] = as_percentage
        result["weighted_judge_correct"] = as_percentage
        result["weighted_both_correct"] = as_percentage
        result["weighted_any_correct"] = as_percentage
        return result
