# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from collections import defaultdict

from nemo_skills.evaluation.metrics.math_metrics import BaseMetrics, as_int, as_percentage


class OmniMetrics(BaseMetrics):
    def __init__(self, compute_no_answer: bool = True, answer_key: str = "generation"):
        super().__init__(compute_no_answer=compute_no_answer)
        self.answer_key = answer_key

    # use same RM code as MathMetrics
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
        if "judgement" in prediction:
            judgement = prediction["judgement"].strip()
            correctness_dict["judge_correct"] = int(judgement.lower() == "a")
            correctness_dict["judge_incorrect"] = -int(
                judgement.lower() == "b"
            )  # negate incorrect so pass@k minimizes it
            correctness_dict["judge_partially_correct"] = int(judgement.lower() == "c")
            correctness_dict["judge_abstained"] = int(judgement.lower() == "d")

            # keep counter of all entries where no answer is correct
            correctness_dict["non_correct"] = -int(judgement.lower() in ["b", "c", "d"])
        return correctness_dict

    def get_metrics(self):
        metrics = super().get_metrics()

        for agg_method, agg_metric_dict in metrics.items():
            correct, incorrect, part_correct, abstained = (
                agg_metric_dict["judge_correct"],
                -agg_metric_dict["judge_incorrect"],  # multiply negated judge_incorrect to get minimized incorrect pct
                agg_metric_dict["judge_partially_correct"],
                agg_metric_dict["judge_abstained"],
            )
            non_correct = -agg_metric_dict["non_correct"]

            # convert pcts back to counts
            if isinstance(correct, float):
                correct *= self.total / 100
            if isinstance(incorrect, float):
                incorrect *= self.total / 100
            if isinstance(part_correct, float):
                part_correct *= self.total / 100
            if isinstance(abstained, float):
                abstained *= self.total / 100
            if isinstance(non_correct, float):
                non_correct *= self.total / 100

            # compute omni index between max correct and min incorrect (for pass@k)
            metrics[agg_method]["judge_omni_index"] = 100 * (correct - incorrect) / self.total if self.total > 0 else 0

            # compute hallucination rate with min incorrect and min non_correct
            metrics[agg_method]["judge_omni_hallucination"] = 100 * incorrect / non_correct if non_correct > 0 else 0
        return metrics

    def get_incorrect_sample(self, prediction: dict) -> dict:
        copy_prediction = prediction.copy()
        if "judgement" in prediction:
            copy_prediction["judgement"] = "B"
            copy_prediction["judge_correct"] = 0
            copy_prediction["judge_incorrect"] = -1
            copy_prediction["judge_partially_correct"] = 0
            copy_prediction["judge_abstained"] = 0
            copy_prediction["non_correct"] = -1
        return copy_prediction

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions, None)
        if "reward_model_score" in predictions[0]:
            self._compute_reward_at_k(predictions=predictions)

    # print the same evaluations/metrics as math but ignoring majority/rm since that doesn't really exist with omniscience
    def evaluations_to_print(self):
        return [
            f"pass@1[avg-of-{self.max_k}]",
            f"pass@{self.max_k}",
        ]

    def metrics_to_print(self):
        metrics_to_print = {
            "num_entries": as_int,
            "avg_tokens": as_int,
            "gen_seconds": as_int,
            "judge_correct": as_percentage,
            "judge_omni_index": as_percentage,
            "judge_omni_hallucination": as_percentage,
        }
        if self.compute_no_answer:
            metrics_to_print["no_answer"] = as_percentage
        return metrics_to_print
