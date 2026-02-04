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

import logging
import re

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_float, as_int, as_percentage
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


class GradingBenchMetrics(BaseMetrics):
    """Metrics for IMO GradingBench evaluation.

    Computes:
    - exact_accuracy: Percentage of exact grade matches
    - binarized_accuracy: Percentage where both grades fall in same bucket
      (correct/almost vs partial/incorrect)
    - mae: Mean Absolute Error between predicted and expected numeric scores

    Grade to score mapping:
    - correct: 7
    - almost: 6
    - partial: 1
    - incorrect: 0
    """

    GRADE_TO_SCORE = {
        "correct": 7,
        "almost": 6,
        "partial": 1,
        "incorrect": 0,
    }
    GRADE_TO_BINARY = {
        "correct": "high",
        "almost": "high",
        "partial": "low",
        "incorrect": "low",
    }
    VALID_GRADES = set(GRADE_TO_SCORE.keys())

    def __init__(self):
        super().__init__(compute_no_answer=False)
        self.mae_errors = []

    def _extract_grade(self, text: str) -> str | None:
        """Extract grade from the last word of judge output.

        Handles markdown formatting and punctuation.
        Returns None if no valid grade (correct/almost/partial/incorrect) is found.
        """
        if not text or not isinstance(text, str):
            LOG.debug("Cannot extract grade: text is empty or not a string")
            return None

        words = text.strip().split()
        if not words:
            LOG.debug("Cannot extract grade: text contains no words")
            return None

        last_word = words[-1]

        # Strip markdown and punctuation, normalize to lowercase
        last_word = re.sub(r"[*_`.,;:!?()[\]{}]+", "", last_word).lower()

        if last_word not in self.VALID_GRADES:
            LOG.debug(
                "Cannot extract grade: '%s' not in valid grades %s. Text ends with: '...%s'",
                last_word,
                self.VALID_GRADES,
                text[-100:] if len(text) > 100 else text,
            )
            return None

        return last_word

    def _get_grades(self, prediction: dict) -> tuple[str | None, str | None]:
        """Extract predicted and expected grades from a prediction."""
        judgement = prediction.get("judgement", "") or prediction.get("generation", "")
        pred_grade = self._extract_grade(judgement)

        expected_answer_raw = prediction.get("expected_answer", "")
        expected_grade = None
        if expected_answer_raw:
            expected_grade = expected_answer_raw.lower().strip()
            if expected_grade not in self.VALID_GRADES:
                LOG.warning(
                    "Invalid expected_answer '%s' - must be one of %s. Entry will be excluded from evaluation.",
                    expected_answer_raw,
                    self.VALID_GRADES,
                )
                expected_grade = None

        return pred_grade, expected_grade

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        """Return correctness scores for a prediction."""
        pred_grade, expected_grade = self._get_grades(prediction)

        exact_match = pred_grade is not None and pred_grade == expected_grade

        pred_binary = self.GRADE_TO_BINARY.get(pred_grade)
        expected_binary = self.GRADE_TO_BINARY.get(expected_grade)
        binary_match = pred_binary is not None and expected_binary is not None and pred_binary == expected_binary

        return {
            "exact_accuracy": exact_match,
            "binarized_accuracy": binary_match,
        }

    def update(self, predictions):
        """Update metrics with predictions."""
        super().update(predictions)

        for pred in predictions:
            pred_grade, expected_grade = self._get_grades(pred)
            pred_score = self.GRADE_TO_SCORE.get(pred_grade)
            expected_score = self.GRADE_TO_SCORE.get(expected_grade)

            if pred_score is not None and expected_score is not None:
                self.mae_errors.append(abs(pred_score - expected_score))

        self._compute_pass_at_k(predictions=predictions)

    def get_metrics(self):
        """Return metrics including MAE."""
        metrics_dict = super().get_metrics()

        if self.mae_errors:
            mae = sum(self.mae_errors) / len(self.mae_errors)
            for agg_mode in metrics_dict:
                metrics_dict[agg_mode]["mae"] = mae
                metrics_dict[agg_mode]["mae_count"] = len(self.mae_errors)

        return metrics_dict

    def reset(self):
        """Reset all tracked metrics."""
        super().reset()
        self.mae_errors = []

    def metrics_to_print(self):
        return {
            "num_entries": as_int,
            "exact_accuracy": as_percentage,
            "binarized_accuracy": as_percentage,
            "mae": as_float,
        }

    def evaluations_to_print(self):
        return [f"pass@1[avg-of-{self.max_k}]", f"pass@{self.max_k}"]
