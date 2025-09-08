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

import re

from nemo_skills.evaluation.metrics.base import BaseMetrics


class ArenaMetrics(BaseMetrics):
    def __init__(self):
        self.reset()

    def _get_judge_score(self, judgment):
        # adapted from https://github.com/lm-sys/arena-hard-auto/blob/main/gen_judgment.py
        pattern = re.compile("\[\[([AB<>=]+)\]\]")
        matches = pattern.findall(judgment)
        matches = [m for m in matches if m != ""]
        if len(set(matches)) == 0:
            return None
        elif len(set(matches)) == 1:
            return matches[0].strip("\n")
        else:
            return None

    def get_incorrect_sample(self, prediction: dict) -> dict:
        prediction = prediction.copy()
        prediction["judgement-gen-base"] = "Rating: [[A>>B]]"
        prediction["judgement-base-gen"] = "Rating: [[B>>A]]"
        return prediction

    def update(self, predictions):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        # this shouldn't do any heavy calculation, but just read the metric from existing json entry
        # all the heavy lifting should be done in the evaluation script
        super().update(predictions)
        self.scores.append([])
        self.agg_mode = f"pass@{len(predictions)}"
        if len(predictions) > 1:
            judge_scores = [self._get_judge_score(elem["judgement-gen-base"]) for elem in predictions]
            # adding the best score out of all the generations
            possible_scores = ["A>>B", "A>B", "A=B", "B>A", "B>>A"]
            for possible_score in possible_scores:
                # picking the best available score
                if any([score == possible_score for score in judge_scores]):
                    self.scores[-1].append(possible_score)
                    best_id = judge_scores.index(possible_score)
                    self.lengths += predictions[best_id].get("num_generated_tokens", 0)
                    break
            else:
                self.scores[-1].append(None)  # in case judge didn't generate a valid score

            judge_scores = [self._get_judge_score(elem["judgement-base-gen"]) for elem in predictions]
            # second score is grading swapped answers, so we iterate from the end
            for possible_score in possible_scores[::-1]:
                # picking the best available score
                if any([score == possible_score for score in judge_scores]):
                    self.scores[-1].append(possible_score)
                    best_id = judge_scores.index(possible_score)
                    self.lengths += predictions[best_id].get("num_generated_tokens", 0)
                    break
            else:
                self.scores[-1].append(None)  # in case judge didn't generate a valid score
        else:
            self.lengths += predictions[0].get("num_generated_tokens", 0)
            self.scores[-1] = [
                self._get_judge_score(predictions[0]["judgement-gen-base"]),
                self._get_judge_score(predictions[0]["judgement-base-gen"]),
            ]

    def get_metrics(self):
        from nemo_skills.evaluation.evaluator.arena import get_aggregate_score

        metrics = {"num_entries": self.total}
        metrics.update(get_aggregate_score(self.scores))
        metrics_dict = {self.agg_mode: metrics}
        self.update_common_metrics(metrics_dict[self.agg_mode])
        # arena metrics have their own confidence estimation, so not doing std metrics here
        return metrics_dict

    def reset(self):
        super().reset()
        self.scores = []  # list of lists
        self.lengths = 0
        # TODO: the class should support pass@k, but this forces it to report as pass@1.
        #       There is some error here for k>1
        self.agg_mode = "pass@1"
