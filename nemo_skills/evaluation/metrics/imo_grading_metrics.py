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

"""
IMO Grading Metrics for IMO GradingBench.

Task: Given a math problem and a student's solution, predict the grade.
- Model predicts one of 4 categories: 7 (Correct), 6 (Almost), 1 (Partial), 0 (Incorrect)
- Accuracy: Predicted category vs Reward category (expected_answer)
- MAE: Mean Absolute Error as percentage = |Predicted category - actual Points (0-7)| / 7 * 100
  - The golden (best possible) MAE is 3.9% due to the simplified 4-category grading scale
"""

import re
from collections import defaultdict

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_int, as_percentage

# Valid predicted grades
VALID_GRADES = {0, 1, 6, 7}


def extract_boxed_grade(text: str) -> int | None:
    """Extract the grade from \\boxed{N} in the model output."""
    if not text:
        return None

    # Try to find \boxed{N} pattern
    patterns = [
        r"\\boxed\{(\d+)\}",
        r"\\boxed\s*\{(\d+)\}",
        r"boxed\{(\d+)\}",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            # Take the last match (final answer)
            try:
                grade = int(matches[-1])
                if grade in VALID_GRADES:
                    return grade
                # If not a valid grade, return None
                return None
            except ValueError:
                continue

    return None


class IMOGradingMetrics(BaseMetrics):
    """
    Metrics for IMO grading benchmark evaluation.

    Computes:
    - grading_correct: Predicted grade matches expected_answer (category)
    - mae: Mean Absolute Error between predicted grade and actual points
    """

    def __init__(self, compute_no_answer: bool = True):
        super().__init__(compute_no_answer=compute_no_answer)
        self.mae_sum = 0.0
        self.mae_counts = defaultdict(lambda: {"sum": 0.0, "count": 0})

    def reset(self):
        super().reset()
        self.mae_sum = 0.0
        self.mae_counts = defaultdict(lambda: {"sum": 0.0, "count": 0})

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        """
        Returns correctness score for a prediction.

        Uses 'grading_correct' field if available (set by evaluator),
        otherwise extracts from generation and compares with expected_answer.
        """
        correctness_dict = {}

        if "grading_correct" in prediction:
            correctness_dict["grading_correct"] = prediction["grading_correct"]
        else:
            # Extract predicted grade from generation
            generation = prediction.get("generation", "") or ""
            predicted = extract_boxed_grade(generation)
            expected = prediction.get("expected_answer")

            if predicted is not None and expected is not None:
                try:
                    expected_int = int(expected)
                    correctness_dict["grading_correct"] = predicted == expected_int
                except (ValueError, TypeError):
                    correctness_dict["grading_correct"] = False
            else:
                correctness_dict["grading_correct"] = False

        return correctness_dict

    def _compute_mae(self, predictions: list[dict]):
        """Compute MAE for predictions."""
        for pred in predictions:
            generation = pred.get("generation", "") or ""
            predicted = extract_boxed_grade(generation)
            points = pred.get("points")

            if predicted is not None and points is not None:
                try:
                    points_int = int(points)
                    mae = abs(predicted - points_int)
                except (ValueError, TypeError):
                    mae = 7  # Max possible MAE
            else:
                # No prediction, use max MAE
                mae = 7

            self.mae_sum += mae

            # Track per-class MAE
            reward_label = pred.get("reward_label", "unknown")
            self.mae_counts[reward_label]["sum"] += mae
            self.mae_counts[reward_label]["count"] += 1

    def get_incorrect_sample(self, prediction: dict) -> dict:
        """Replace prediction with something that evaluates as incorrect."""
        prediction = prediction.copy()
        prediction["grading_correct"] = False
        prediction["generation"] = ""
        return prediction

    def update(self, predictions):
        """Update evaluation results with current element.

        Args:
            predictions (list[dict]): Aggregated predictions across all generations.
        """
        super().update(predictions)
        predicted_answers = []

        for pred in predictions:
            generation = pred.get("generation", "") or ""
            grade = extract_boxed_grade(generation)
            predicted_answers.append(str(grade) if grade is not None else None)

        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)
        self._compute_mae(predictions)

    def get_metrics(self):
        """Get all computed metrics."""
        metrics_dict = super().get_metrics()

        # Add MAE to all aggregation modes
        # MAE is reported as percentage: |predicted - points| / 7 * 100
        # The golden (best possible) MAE is 3.9% due to simplified 4-category grading
        for agg_mode in metrics_dict:
            if self.total > 0:
                mae_absolute = self.mae_sum / self.total
                # Convert to percentage (divide by max score 7, multiply by 100)
                metrics_dict[agg_mode]["mae"] = (mae_absolute / 7) * 100

        # Add per-class MAE stats
        per_class_mae = {}
        for reward_label, stats in self.mae_counts.items():
            if stats["count"] > 0:
                mae_absolute = stats["sum"] / stats["count"]
                per_class_mae[reward_label] = {
                    "count": stats["count"],
                    # Convert to percentage for consistency
                    "mae": (mae_absolute / 7) * 100,
                }
        if per_class_mae:
            for agg_mode in metrics_dict:
                metrics_dict[agg_mode]["per_class_mae"] = per_class_mae

        return metrics_dict

    def evaluations_to_print(self):
        """Which aggregation modes to print."""
        return [
            f"pass@1[avg-of-{self.max_k}]",
            f"majority@{self.max_k}",
            f"pass@{self.max_k}",
        ]

    def metrics_to_print(self):
        """Which metrics to print and how to format them."""
        return {
            "num_entries": as_int,
            "avg_tokens": as_int,
            "gen_seconds": as_int,
            "grading_correct": as_percentage,
            "mae": as_percentage,  # MAE is now reported as percentage
            "no_answer": as_percentage,
        }
