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
from collections import defaultdict

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

        Each (row, sample) pair contributes its own judgment pair to ``self.scores``,
        so the bootstrap Elo in ``get_aggregate_score`` naturally averages over
        sampling variance — equivalent to ``pass@1[avg-of-N]`` for binary outcomes.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        super().update(predictions)
        category = predictions[0].get("category")
        for pred in predictions:
            self.scores.append(
                [
                    self._get_judge_score(pred["judgement-gen-base"]),
                    self._get_judge_score(pred["judgement-base-gen"]),
                ]
            )
            self.categories.append(category)

    def get_metrics(self):
        from nemo_skills.evaluation.evaluator.arena import get_aggregate_score

        overall_metrics = {"num_entries": self.total}
        overall_metrics.update(get_aggregate_score(self.scores))
        self.update_common_metrics(overall_metrics)

        category_scores = defaultdict(list)
        for score, category in zip(self.scores, self.categories, strict=True):
            category_scores[category].append(score)

        if len(set(self.categories)) > 1:
            for category, scores in category_scores.items():
                cat_metrics = {"num_entries": len(scores)}
                cat_metrics.update(get_aggregate_score(scores))
                overall_metrics[f"category_{category}"] = cat_metrics

        # pass@1 already aggregates every (row, sample) battle pair under task:N>1,
        # so it doubles as pass@1[avg-of-N]; we don't emit a redundant alias.
        # arena metrics have their own bootstrap CI, so not adding std metrics here.
        return {"pass@1": overall_metrics}

    def evaluations_to_print(self):
        return ["pass@1"]

    def reset(self):
        super().reset()
        self.scores = []  # flat list of (gen-base, base-gen) judgment pairs
        self.categories = []  # parallel list of category strings
