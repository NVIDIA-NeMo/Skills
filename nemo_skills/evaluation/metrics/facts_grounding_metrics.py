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


class FactsGroundingMetrics(BaseMetrics):
    """Metrics for the Google FACTS Grounding benchmark.

    Tracks three score channels:
    - grounding_correct: whether the response is fully grounded in the context
    - quality_passed: whether the response adequately addresses the user request
    - factuality_correct: grounding_correct AND quality_passed (the headline metric)
    """

    def __init__(self, compute_no_answer: bool = False):
        super().__init__(compute_no_answer=compute_no_answer)

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        grounding = bool(prediction.get("judgement_grounding", False))
        quality = bool(prediction.get("judgement_quality", True))
        return {
            "grounding_correct": grounding,
            "quality_passed": quality,
            "factuality_correct": grounding and quality,
        }

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions=predictions)
