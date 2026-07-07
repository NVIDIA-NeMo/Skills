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

import logging

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_float, as_int, as_percentage
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


class SpecdecMetrics(BaseMetrics):
    """Metrics for SPEED-Bench speculative decoding evaluation.

    Reads per-data-point speculative decoding statistics that were computed
    by the generation task using a before/after delta of the server's
    Prometheus ``/metrics`` counters and stamped by the ``eval_specdec``
    evaluator.

    Key metrics (computed from Prometheus counter deltas):

    * **acceptance_length** (AL) — ``1 + delta_accepted / delta_drafts``.
      The "+1" accounts for the target model's verified token that is always
      emitted even on a full rejection.
    * **acceptance_rate** — ``(delta_accepted / delta_draft_tokens) * 100``.
    * **per_position_acceptance_rates** — per-position acceptance probability,
      computed as ``delta_per_pos[i] / delta_drafts``.

    """

    def __init__(self):
        super().__init__(compute_no_answer=False)

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        return {
            "spec_draft_tokens": prediction["draft_tokens"],
            "spec_accepted_tokens": prediction["accepted_tokens"],
            "spec_num_drafts": prediction["num_drafts"],
            "spec_acceptance_length": prediction["acceptance_length"],
            "spec_acceptance_rate": prediction["acceptance_rate"],
        }

    def update(self, predictions: list[dict]) -> None:
        """Update the evaluation results with the current element.

        Args:
            predictions: Aggregated predictions across all generations.
                Each prediction should contain speculative decoding metrics
                stamped by the ``eval_specdec`` evaluator.
        """
        super().update(predictions)
        self._compute_pass_at_k(
            predictions=predictions, predicted_answers=[pred.get("generation", None) for pred in predictions]
        )

    def get_metrics(self) -> dict:
        """Get all computed metrics including speculative decoding statistics.

        Returns:
            Nested dict of evaluation mode → metric name → value.
        """
        metrics_dict = {}

        for agg_mode, agg_metric_dict in self.eval_dict.items():
            metrics_dict[agg_mode] = {}
            self.update_common_metrics(metrics_dict[agg_mode])
            for metric_key, metric_value in agg_metric_dict.items():
                if metric_key.startswith("spec_"):
                    metrics_dict[agg_mode][metric_key] = metric_value / self.total
                else:
                    metrics_dict[agg_mode][metric_key] = metric_value
        self._add_std_metrics(metrics_dict)

        return metrics_dict

    def metrics_to_print(self) -> dict:
        """Control which metrics are displayed in the summary table."""
        metrics_to_print = {
            "num_entries": as_int,
            "avg_tokens": as_int,
            "gen_seconds": as_int,
            "spec_acceptance_length": as_float,
            "spec_acceptance_rate": as_float,
        }
        if self.compute_no_answer:
            metrics_to_print["no_answer"] = as_percentage
        return metrics_to_print
