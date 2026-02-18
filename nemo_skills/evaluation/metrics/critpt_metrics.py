# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_int, as_percentage
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


class CritPtMetrics(BaseMetrics):
    """Metrics computation for CritPt benchmark.

    CritPt evaluation is done via external API which returns aggregate accuracy.
    The evaluator distributes correctness by marking the first n examples as correct,
    where n = int(accuracy * total_examples).

    Note: This is a limitation of the CritPt API which doesn't provide per-example results.
    """

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        """Extract score from CritPt evaluation results.

        The evaluator sets is_correct based on aggregate accuracy distribution.
        Since CritPt API doesn't provide per-example correctness, the first n examples
        are marked as correct where n = int(aggregate_accuracy * total_examples).
        """
        return {"accuracy": prediction["full_dataset_accuracy"]}

    def update(self, predictions):
        """Update metrics with new predictions.

        Args:
            predictions: List of prediction dictionaries with accuracy field.
        """
        super().update(predictions)
        self._compute_pass_at_k(predictions=predictions)

    def metrics_to_print(self):
        """Specify which metrics to print and their formatting."""
        return {
            "num_entries": as_int,
            "avg_tokens": as_int,
            "gen_seconds": as_int,
            "accuracy": as_percentage,
        }
