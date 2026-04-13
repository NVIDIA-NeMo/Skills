# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Metrics aggregation for ContextASR-Bench evaluation.

Computes corpus-level:
- WER: total word errors / total reference words
- NE-WER: total entity word errors / total entity reference words
- NE-FNR: 1 - (total entity hits / total entities)
"""

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_int, as_percentage


class ContextASRMetrics(BaseMetrics):
    """Metrics class for ContextASR-Bench with corpus-level WER, NE-WER, NE-FNR."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer)
        self.max_k = max_k

        # Corpus-level WER accumulators
        self.wer_total_errors = 0
        self.wer_total_ref_words = 0

        # Corpus-level NE-WER accumulators
        self.ne_wer_total_errors = 0
        self.ne_wer_total_ref_words = 0

        # Corpus-level NE-FNR accumulators
        self.ne_fnr_total_hits = 0
        self.ne_fnr_total_entities = 0

    def _get_score_dict(self, prediction):
        return {"is_correct": prediction.get("is_correct", False)}

    def get_incorrect_sample(self, prediction):
        prediction = prediction.copy()
        prediction["is_correct"] = False
        return prediction

    def update_common_metrics(self, agg_dict):
        agg_dict["num_entries"] = self.total
        agg_dict["avg_tokens"] = int(self.avg_tokens / self.total) if self.total > 0 else 0
        if self.max_end_time > float("-inf") and self.min_start_time < float("inf"):
            agg_dict["gen_seconds"] = int(self.max_end_time - self.min_start_time)

    def update(self, predictions):
        super().update(predictions)

        predicted_answers = [pred.get("generation", "").strip() or None for pred in predictions]

        for pred in predictions:
            self.wer_total_errors += pred.get("wer_errors", 0)
            self.wer_total_ref_words += pred.get("wer_ref_words", 0)
            self.ne_wer_total_errors += pred.get("ne_wer_errors", 0)
            self.ne_wer_total_ref_words += pred.get("ne_wer_ref_words", 0)
            self.ne_fnr_total_hits += pred.get("ne_fnr_hits", 0)
            self.ne_fnr_total_entities += pred.get("ne_fnr_total", 0)

        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)

    def get_metrics(self):
        metrics_dict = super().get_metrics()

        for _agg_mode, agg_metrics in metrics_dict.items():
            if "correct" in agg_metrics:
                agg_metrics["success_rate"] = agg_metrics["correct"]

            if self.wer_total_ref_words > 0:
                agg_metrics["wer"] = round(
                    100.0 * self.wer_total_errors / self.wer_total_ref_words, 2
                )

            if self.ne_wer_total_ref_words > 0:
                agg_metrics["ne_wer"] = round(
                    100.0 * self.ne_wer_total_errors / self.ne_wer_total_ref_words, 2
                )

            if self.ne_fnr_total_entities > 0:
                agg_metrics["ne_fnr"] = round(
                    100.0 * (1.0 - self.ne_fnr_total_hits / self.ne_fnr_total_entities), 2
                )

        return metrics_dict

    def evaluations_to_print(self):
        evals = [f"pass@{self.max_k}"]
        if self.max_k > 1:
            evals.extend([f"majority@{self.max_k}", f"pass@1[avg-of-{self.max_k}]"])
        return evals

    def metrics_to_print(self):
        base_metrics = {
            "avg_tokens": as_int,
            "gen_seconds": as_int,
            "success_rate": as_percentage,
        }

        if self.compute_no_answer:
            base_metrics["no_answer"] = as_percentage

        if self.wer_total_ref_words > 0:
            base_metrics["wer"] = as_percentage
        if self.ne_wer_total_ref_words > 0:
            base_metrics["ne_wer"] = as_percentage
        if self.ne_fnr_total_entities > 0:
            base_metrics["ne_fnr"] = as_percentage

        base_metrics["num_entries"] = as_int
        return base_metrics
