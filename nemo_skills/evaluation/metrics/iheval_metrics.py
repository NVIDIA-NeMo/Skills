# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import math
from collections import defaultdict

from nemo_skills.evaluation.metrics.base import BaseMetrics


class IHEvalMetrics(BaseMetrics):
    """Pass@k over symbolic_correct with per-setting / per-variant breakdowns."""

    def _get_score_dict(self, prediction: dict) -> dict[str, float]:
        return {"symbolic_correct": float(prediction.get("symbolic_correct", 0.0))}

    def get_incorrect_sample(self, prediction: dict) -> dict:
        prediction = prediction.copy()
        prediction["symbolic_correct"] = 0.0
        return prediction

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions=predictions)
        self._sample_scores.append([float(p.get("symbolic_correct", 0.0)) for p in predictions])
        self._sample_settings.append(predictions[0].get("setting"))
        self._sample_variants.append(predictions[0].get("variant"))

    def get_metrics(self):
        metrics_dict = super().get_metrics()
        for mode_key, agg_dict in metrics_dict.items():
            aggregator = self._mode_aggregator(mode_key)
            if aggregator is None:
                continue
            setting_sums: dict[str, float] = defaultdict(float)
            setting_counts: dict[str, int] = defaultdict(int)
            variant_sums: dict[str, float] = defaultdict(float)
            variant_counts: dict[str, int] = defaultdict(int)
            for scores, setting, variant in zip(self._sample_scores, self._sample_settings, self._sample_variants):
                value = aggregator(scores)
                if setting:
                    setting_sums[setting] += value
                    setting_counts[setting] += 1
                if variant:
                    variant_sums[variant] += value
                    variant_counts[variant] += 1
            for setting, count in setting_counts.items():
                agg_dict[f"setting_{setting}"] = 100.0 * setting_sums[setting] / count
            for variant, count in variant_counts.items():
                agg_dict[f"variant_{variant}"] = 100.0 * variant_sums[variant] / count
        return metrics_dict

    @staticmethod
    def _mode_aggregator(mode_key: str):
        # Mirror base._compute_pass_at_k per mode so breakdowns stay consistent with
        # symbolic_correct. pass@1 has no special case on purpose: base uses the same
        # probabilistic formula at k=1 for binary scores, which _pass_at_k reproduces.
        if mode_key.startswith("pass@1[avg-of-") and mode_key.endswith("]"):
            try:
                k = int(mode_key[len("pass@1[avg-of-") : -1])
            except ValueError:
                return None
            return lambda scores: sum(scores[:k]) / k if scores else 0.0
        if mode_key.startswith("pass@"):
            try:
                k = int(mode_key[len("pass@") :])
            except ValueError:
                return None
            return lambda scores: _pass_at_k(scores, k)
        return None

    def reset(self):
        super().reset()
        self._sample_scores: list[list[float]] = []
        self._sample_settings: list[str | None] = []
        self._sample_variants: list[str | None] = []


def _pass_at_k(scores: list[float], k: int) -> float:
    if not scores:
        return 0.0
    if all(s in (0, 1, True, False) for s in scores):
        total = len(scores)
        total_incorrect = total - int(sum(scores))
        if total_incorrect < k:
            return 1.0
        return 1.0 - math.comb(total_incorrect, k) / math.comb(total, k)
    return max(scores[:k])
