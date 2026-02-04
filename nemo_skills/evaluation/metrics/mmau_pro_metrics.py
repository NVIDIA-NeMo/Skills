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
import re

import numpy as np

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_int, as_percentage
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def extract_multicriteria_scores(judgement_text: str) -> dict[str, float]:
    """Extract multi-criteria scores (1-5 scale) from LLM judge evaluation.

    Expected format:
        CORRECTNESS: [score] - [justification]
        RELEVANCE: [score] - [justification]
        COMPLETENESS: [score] - [justification]
        CLARITY: [score] - [justification]
        OVERALL: [score] - [overall assessment]

    Returns:
        Dictionary with keys: correctness, relevance, completeness, clarity, overall
        Defaults to 3.0 if score not found.
    """
    scores = {}
    found_overall = False

    patterns = {
        "correctness": r"CORRECTNESS:\s*(\d+(?:\.\d+)?)",
        "relevance": r"RELEVANCE:\s*(\d+(?:\.\d+)?)",
        "completeness": r"COMPLETENESS:\s*(\d+(?:\.\d+)?)",
        "clarity": r"CLARITY:\s*(\d+(?:\.\d+)?)",
        "overall": r"OVERALL:\s*(\d+(?:\.\d+)?)",
    }

    for criterion, pattern in patterns.items():
        match = re.search(pattern, judgement_text, re.IGNORECASE)
        if match:
            scores[criterion] = float(match.group(1))
            if criterion == "overall":
                found_overall = True
        else:
            scores[criterion] = 3.0

    # Fallback: compute overall only if not explicitly provided by judge
    if not found_overall:
        criteria_scores = [scores.get(k, 3.0) for k in ["correctness", "relevance", "completeness", "clarity"]]
        scores["overall"] = sum(criteria_scores) / len(criteria_scores)

    return scores


class MMAUProMetrics(BaseMetrics):
    """Metrics class for MMAU-Pro benchmark (all subgroups)."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer)
        self.max_k = max_k

        # Track multi-criteria scores for open-ended questions (1-5 scale)
        self.multicriteria_scores = {
            "correctness": [],
            "relevance": [],
            "completeness": [],
            "clarity": [],
            "overall": [],
        }

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        """Extract correctness scores from prediction."""
        score_dict = {}

        # Open-ended: use LLM judge correctness score >= 3 as correct
        if "judgement" in prediction:
            multicriteria = extract_multicriteria_scores(prediction["judgement"])
            score_dict["correct"] = multicriteria.get("correctness", 3.0) >= 3.0
        # Closed-form / instruction-following: use binary correctness
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

        predicted_answers = [(pred.get("generation") or "").strip() or None for pred in predictions]
        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)

        # Collect multi-criteria scores for open-ended questions
        for pred in predictions:
            if "judgement" in pred:
                multicriteria = extract_multicriteria_scores(pred["judgement"])
                for criterion in self.multicriteria_scores:
                    self.multicriteria_scores[criterion].append(multicriteria.get(criterion, 3.0))

    def get_metrics(self):
        """Get computed metrics."""
        metrics_dict = super().get_metrics()

        for agg_mode, agg_metrics in metrics_dict.items():
            # Ensure avg_tokens is present
            if "avg_tokens" not in agg_metrics:
                agg_metrics["avg_tokens"] = 0
            if "no_answer" in agg_metrics:
                agg_metrics["no_answer"] = agg_metrics["no_answer"] / 2.0

            # Add multi-criteria averages for open-ended (convert 1-5 scale to percentage)
            if self.multicriteria_scores["overall"]:
                for criterion in self.multicriteria_scores:
                    scores = self.multicriteria_scores[criterion]
                    if scores:
                        # Convert 1-5 scale to 0-100 percentage scale
                        avg_score = np.mean(scores)
                        std_score = np.std(scores)
                        agg_metrics[f"avg_{criterion}"] = (avg_score / 5.0) * 100
                        agg_metrics[f"std_{criterion}"] = (std_score / 5.0) * 100

                # Set correct and success_rate to avg_correctness for open-ended
                agg_metrics["correct"] = agg_metrics["avg_correctness"]
                agg_metrics["success_rate"] = agg_metrics["avg_correctness"]

                # Calculate good/poor response rates based on overall >= 4 or <= 2
                overall_scores = self.multicriteria_scores["overall"]
                good_responses = sum(1 for score in overall_scores if score >= 4.0)
                poor_responses = sum(1 for score in overall_scores if score <= 2.0)

                agg_metrics["good_response_rate"] = (good_responses / len(overall_scores)) * 100
                agg_metrics["poor_response_rate"] = (poor_responses / len(overall_scores)) * 100

            # For closed-form / instruction-following: use binary correctness
            elif "correct" in agg_metrics:
                agg_metrics["success_rate"] = agg_metrics["correct"]

            # Round all numeric values to 2 decimal places
            for key, value in agg_metrics.items():
                if isinstance(value, float) and not isinstance(value, bool):
                    agg_metrics[key] = round(value, 2)

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

        # Add multi-criteria metrics for open-ended questions (now in percentage format)
        if self.multicriteria_scores["overall"]:
            base_metrics.update(
                {
                    "avg_overall": as_percentage,
                    "avg_correctness": as_percentage,
                    "avg_relevance": as_percentage,
                    "avg_completeness": as_percentage,
                    "avg_clarity": as_percentage,
                    "good_response_rate": as_percentage,
                    "poor_response_rate": as_percentage,
                }
            )

        base_metrics["num_entries"] = as_int
        return base_metrics
