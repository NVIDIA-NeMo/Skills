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

from nemo_skills.evaluation.metrics.base import BaseMetrics


class GraphWalksMetrics(BaseMetrics):
    """Metrics for the GraphWalks benchmark (openai/graphwalks).

    Reads pre-computed fields written by eval_graphwalks in the evaluator.

    Metrics reported:
    - ``f1``: set-level F1 between predicted and expected node sets.
    - ``parse_failed``: 1.0 iff the parse failed.
    """

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        return {"f1": prediction["f1"], "parse_failed": prediction["parse_failed"]}

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions=predictions)
