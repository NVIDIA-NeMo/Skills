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

import logging

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_int, as_percentage
from nemo_skills.evaluation.metrics.utils import is_correct_judgement
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


class MMAUProMetrics(BaseMetrics):
    """Metrics class for MMAU-Pro benchmark (all subgroups)."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer)
        self.max_k = max_k

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        """Extract correctness scores from prediction."""
        score_dict = {}

        # Open-ended: extract from judge result
        if "judgement" in prediction:
            judge_result = is_correct_judgement(prediction["judgement"])
            score_dict["judge_correct"] = judge_result
            score_dict["correct"] = judge_result
        # Closed-form and instruction following: use is_correct
        elif "is_correct" in prediction:
            score_dict["correct"] = prediction["is_correct"]
        else:
            score_dict["correct"] = False

        return score_dict

    def get_incorrect_sample(self, prediction: dict) -> dict:
        """Return a sample marked as incorrect."""
        prediction = prediction.copy()
        prediction["is_correct"] = False
        if "judgement" in prediction:
            prediction["judge_correct"] = False
        if not prediction.get("generation", "").strip():
            prediction["generation"] = None
        return prediction

    def update(self, predictions):
        """Update metrics with new predictions."""
        super().update(predictions)
        predicted_answers = [pred.get("generation", None).strip() or None for pred in predictions]
        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)

    def get_metrics(self):
        """Get computed metrics."""
        metrics_dict = super().get_metrics()
        for agg_mode, agg_metrics in metrics_dict.items():
            # Ensure avg_tokens is always present for MMAU-Pro
            if "avg_tokens" not in agg_metrics:
                agg_metrics["avg_tokens"] = 0
            if "no_answer" in agg_metrics:
                agg_metrics["no_answer"] = agg_metrics["no_answer"] / 2.0
            # Set success_rate from correct or judge_correct
            if "judge_correct" in agg_metrics:
                agg_metrics["success_rate"] = agg_metrics["judge_correct"]
            elif "correct" in agg_metrics:
                agg_metrics["success_rate"] = agg_metrics["correct"]
        return metrics_dict

    def metrics_to_print(self):
        """Specify which metrics to print."""
        base_metrics = {
            "avg_tokens": as_int,
            "gen_seconds": as_int,
            "success_rate": as_percentage,
        }
        if self.compute_no_answer:
            base_metrics["no_answer"] = as_percentage
        base_metrics["num_entries"] = as_int
        return base_metrics
