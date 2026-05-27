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

from collections import defaultdict

from nemo_skills.evaluation.metrics.base import BaseMetrics


class IHEvalMetrics(BaseMetrics):
    """Metrics for IHEval sub-benchmarks.

    Uses the base ``_compute_pass_at_k`` machinery so that running ``iheval:N``
    (N samples per question) yields proper ``pass@1[avg-of-N]`` / ``pass@N``
    aggregates over the per-row ``symbolic_correct`` score (which may be
    fractional, e.g. F1/ROUGE-L for the task-execution subs). On top of that it
    tracks per-``setting`` (aligned/conflict/reference) and per-``variant``
    breakdowns, averaged over attempts so they remain meaningful for pass@k runs
    (and match pass@1 when N=1).
    """

    def _get_score_dict(self, prediction: dict) -> dict[str, float]:
        return {"symbolic_correct": float(prediction.get("symbolic_correct", 0.0))}

    def get_incorrect_sample(self, prediction: dict) -> dict:
        prediction = prediction.copy()
        prediction["symbolic_correct"] = 0.0
        return prediction

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions=predictions)

        # Per-setting / per-variant tracking uses avg-over-attempts so it still
        # works for pass@k-style sampling runs, while matching pass@1 when k=1.
        scores = [float(p.get("symbolic_correct", 0.0)) for p in predictions]
        avg = sum(scores) / len(scores)

        setting = predictions[0].get("setting")
        if setting:
            self._setting_totals[setting] += 1
            self._setting_correct[setting] += avg

        variant = predictions[0].get("variant")
        if variant:
            self._variant_totals[variant] += 1
            self._variant_correct[variant] += avg

    def get_metrics(self):
        metrics_dict = super().get_metrics()
        for agg_dict in metrics_dict.values():
            for setting, total in self._setting_totals.items():
                if total > 0:
                    agg_dict[f"setting_{setting}"] = 100.0 * self._setting_correct[setting] / total
            for variant, total in self._variant_totals.items():
                if total > 0:
                    agg_dict[f"variant_{variant}"] = 100.0 * self._variant_correct[variant] / total
        return metrics_dict

    def reset(self):
        super().reset()
        self._setting_totals: dict[str, int] = defaultdict(int)
        self._setting_correct: dict[str, float] = defaultdict(float)
        self._variant_totals: dict[str, int] = defaultdict(int)
        self._variant_correct: dict[str, float] = defaultdict(float)
