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

import logging
from collections import defaultdict

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_int, as_percentage
from nemo_skills.evaluation.metrics.utils import is_correct_judgement
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


class MathMetrics(BaseMetrics):
    # TODO: how can we ensure that user-defined aggregations have all the same metrics as in base?

    def __init__(
        self, compute_no_answer: bool = True, question_key: str = "problem", answer_key: str = "predicted_answer"
    ):
        super().__init__(compute_no_answer=compute_no_answer)
        self.question_key = question_key
        self.answer_key = answer_key

    def _compute_reward_at_k(self, predictions: list[dict]):
        score_dicts = [self._get_score_dict(pred) for pred in predictions]

        for k in range(1, len(predictions) + 1):
            for score_method in score_dicts[0].keys():
                # Get valid answers and their results for this field
                valid_answers_and_results = [
                    (elem[self.answer_key], correctness_dict[score_method], elem["reward_model_score"])
                    for elem, correctness_dict in zip(predictions[:k], score_dicts[:k])
                    if elem[self.answer_key] is not None
                ]

                # If no valid answers, it's incorrect
                if not valid_answers_and_results:
                    is_correct = False
                else:
                    is_correct_best = sorted(valid_answers_and_results, key=lambda x: x[2], reverse=True)[0][1]
                    self.eval_dict[f"rm_best@{k}"][score_method] += is_correct_best

                    answer_to_score_dict = defaultdict(float)
                    answer_to_correctness_dict = {}
                    for predicted_answer, is_correct, reward_score in valid_answers_and_results:
                        answer_to_score_dict[predicted_answer] += reward_score
                        answer_to_correctness_dict[predicted_answer] = is_correct

                    top_cum_reward_answer = sorted(
                        list(answer_to_score_dict.items()), key=lambda x: x[1], reverse=True
                    )[0][0]
                    is_correct_majority = answer_to_correctness_dict[top_cum_reward_answer]
                    self.eval_dict[f"rm_majority@{k}"][score_method] += is_correct_majority

            no_answer = all(elem[self.answer_key] is None for elem in predictions[:k])
            self.eval_dict[f"rm_best@{k}"]["no_answer"] += no_answer
            self.eval_dict[f"rm_majority@{k}"]["no_answer"] += no_answer

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        correctness_dict = {}
        if "symbolic_correct" in prediction:
            correctness_dict["symbolic_correct"] = prediction["symbolic_correct"]
        if "judgement" in prediction:
            correctness_dict["judge_correct"] = is_correct_judgement(prediction["judgement"])
        if "judge_correct" in correctness_dict and "symbolic_correct" in correctness_dict:
            correctness_dict["both_correct"] = (
                correctness_dict["symbolic_correct"] and correctness_dict["judge_correct"]
            )
            correctness_dict["any_correct"] = correctness_dict["symbolic_correct"] or correctness_dict["judge_correct"]

        return correctness_dict

    def get_incorrect_sample(self, prediction: dict) -> dict:
        prediction = prediction.copy()
        if "symbolic_correct" in prediction:
            prediction["symbolic_correct"] = False
        if "judgement" in prediction:
            prediction["judgement"] = "Judgement: No"
        prediction[self.answer_key] = None
        return prediction

    def update(self, predictions):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        super().update(predictions)
        predicted_answers = [pred[self.answer_key] for pred in predictions]
        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)

        if "reward_model_score" in predictions[0]:
            self._compute_reward_at_k(predictions=predictions)

        # Log discrepancies between the two judgements
        for prediction in predictions:
            correctness_dict = self._get_score_dict(prediction)
            if "symbolic_correct" not in correctness_dict or "judge_correct" not in correctness_dict:
                continue
            if correctness_dict["symbolic_correct"] != correctness_dict["judge_correct"]:
                LOG.debug(
                    "Discrepancy between symbolic (%s) and LLM checkers (%s).\n"
                    "Question: %s\nPredicted answer: %s\nExpected answer: %s\nLLM reasoning: %s\n",
                    correctness_dict["symbolic_correct"],
                    correctness_dict["judge_correct"],
                    prediction[self.question_key],
                    prediction[self.answer_key],
                    prediction["expected_answer"],
                    prediction["judgement"],
                )

    def evaluations_to_print(self):
        """We will log all majority/rm/pass/pass@1[avg-of-k] up to k, but only report the kth one."""
        return [
            f"pass@1[avg-of-{self.max_k}]",
            f"majority@{self.max_k}",
            f"rm_best@{self.max_k}",
            f"rm_majority@{self.max_k}",
            f"pass@{self.max_k}",
        ]

    def metrics_to_print(self):
        metrics_to_print = {
            "num_entries": as_int,
            "avg_tokens": as_int,
            "gen_seconds": as_int,
            "judge_correct": as_percentage,
            "symbolic_correct": as_percentage,
        }
        if self.compute_no_answer:
            metrics_to_print["no_answer"] = as_percentage
        return metrics_to_print
