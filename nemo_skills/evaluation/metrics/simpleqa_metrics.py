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

# Code adapted from https://www.kaggle.com/code/nanliao7/simpleqa-verified-benchmark-starter-code

from nemo_skills.evaluation.metrics.base import BaseMetrics

CORRECT_LABEL = "A"
INCORRECT_LABEL = "B"
NOT_ATTEMPTED_LABEL = "C"


def is_correct_judgement_label_matching(judgement: str, correct_label: str) -> bool:
    """Check if judgement label matches the correct label.
    For example, if the correct label is "A", then "A" or "a" will be considered correct.

    For reference, SimpleQA judge returns: A: CORRECT, B: INCORRECT, C: NOT_ATTEMPTED
    """
    if not judgement:
        return False
    judgement = judgement.strip()
    if judgement == correct_label or judgement[0] == correct_label:
        return True
    return False


class SimpleQAMetrics(BaseMetrics):
    """Metrics for SimpleQA with F1 for pass@k and pass@1[avg-of-k]."""

    def __init__(self, compute_no_answer: bool = False, answer_key: str = "predicted_answer"):
        super().__init__(compute_no_answer=compute_no_answer)
        self.answer_key = answer_key

    def update(self, predictions):
        """Update evaluation results with current predictions.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                Each prediction should contain 'judgement' field with judge evaluation.
        """
        super().update(predictions)

        predicted_answers = [pred["judgement"] for pred in predictions]

        self._compute_pass_at_k(predictions=predictions)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)

    # --- required for BaseMetrics to compute pass/majority ---
    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        """
        Returns correctness channels; "correct" is the primary pass@k signal.
        """
        d = prediction
        out = {
            "correct": False,
            "incorrect": False,
            "not_attempted": False,
        }
        judgement = d.get("judgement", None)
        if judgement is None:
            out["not_attempted"] = True
        else:
            is_corr = is_correct_judgement_label_matching(judgement, CORRECT_LABEL)
            is_inc = is_correct_judgement_label_matching(judgement, INCORRECT_LABEL)
            is_na = is_correct_judgement_label_matching(judgement, NOT_ATTEMPTED_LABEL)
            out["correct"] = bool(is_corr)
            out["incorrect"] = bool(is_inc)
            out["not_attempted"] = bool(is_na)
            if not any([is_corr, is_inc, is_na]):
                # Following https://www.kaggle.com/code/nanliao7/simpleqa-verified-benchmark-starter-code
                # If the judgement is unparseable, we consider it as not attempted
                # DEFAULT_GRADE_IF_UNPARSEABLE = "C"  # Corresponds to NOT_ATTEMPTED
                out["not_attempted"] = True

        return out

    # --- helpers for F1 bookkeeping ---
    @staticmethod
    def _to_bool_or_none(j):
        """Map a judgement label to True/False/None using your helper."""
        # True=correct, False=incorrect, None=not attempted
        if is_correct_judgement_label_matching(j, CORRECT_LABEL):
            return True
        elif is_correct_judgement_label_matching(j, INCORRECT_LABEL):
            return False
        elif is_correct_judgement_label_matching(j, NOT_ATTEMPTED_LABEL):
            return None
        else:
            return None

    # --- derive P/R/F1 after BaseMetrics aggregates the rest ---
    def get_metrics(self):
        metrics = super().get_metrics()  # adds % scaling for float metrics, leaves ints as ints

        def cal_f1(agg_bucket: dict):
            correct = agg_bucket.get("correct", 0) / 100
            incorrect = agg_bucket.get("incorrect", 0) / 100

            is_given_attempted = correct + incorrect
            accuracy_given_attempted = correct / is_given_attempted if is_given_attempted > 0 else 0.0
            f1 = (
                2 * accuracy_given_attempted * correct / (accuracy_given_attempted + correct)
                if (accuracy_given_attempted + correct) > 0
                else 0.0
            )
            agg_bucket["f1"] = f1 * 100
            agg_bucket["accuracy_given_attempted"] = accuracy_given_attempted * 100

        # Attach PRF1 to all relevant groups produced during evaluation
        for group_name, bucket in metrics.items():
            cal_f1(bucket)

        return metrics
