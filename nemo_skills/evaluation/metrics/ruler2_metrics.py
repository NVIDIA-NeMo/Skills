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


class Ruler2Metrics(BaseMetrics):
    """Metrics for RULER v2 benchmarks.

    RULER v2 tasks use two different correctness keys depending on the eval type:
    - ``is_correct`` (float 0-1): soft string-match score used by tasks with
      ``eval_type=ruler2`` (NIAH and QA variants).
    - ``symbolic_correct`` (bool): exact-match score used by tasks with
      ``eval_type=multichoice`` (mk_niah_hard, mk_niah_medium).
    """

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        if "is_correct" in prediction:
            return {"accuracy": prediction["is_correct"]}
        return {"accuracy": prediction["symbolic_correct"]}

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions=predictions)

    def get_incorrect_sample(self, prediction: dict) -> dict:
        prediction = prediction.copy()
        if "is_correct" in prediction:
            prediction["is_correct"] = False
        elif "symbolic_correct" in prediction:
            prediction["symbolic_correct"] = False
        else:
            raise ValueError(f"expected 'is_correct' or 'symbolic_correct' to be in keys: {list(prediction.keys())=}")
        return prediction
