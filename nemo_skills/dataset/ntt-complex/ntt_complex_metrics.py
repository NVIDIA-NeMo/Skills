# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Metrics and group scoring for NTT-COMPLEX."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from nemo_skills.evaluation.metrics.audio_metrics import AudioMetrics
from nemo_skills.evaluation.metrics.base import as_percentage


class NTTComplexMetrics(AudioMetrics):
    """AudioMetrics extension with strict-format success signals."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer, max_k=max_k)
        self.format_valid = 0
        self.format_total = 0
        self.format_ast_correct = 0
        self.by_format: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def update(self, predictions):
        for pred in predictions:
            if "format_valid" in pred:
                self.format_total += 1
                self.format_valid += int(bool(pred["format_valid"]))
                self.format_ast_correct += int(bool(pred.get("format_ast_is_correct")))
                self.by_format[str(pred.get("format_id", "unknown"))].append(pred)
        super().update(predictions)

    def _format_metrics(self) -> dict[str, float]:
        if self.format_total <= 0:
            return {}
        metrics: dict[str, float] = {
            "format_valid_rate": round(100.0 * self.format_valid / self.format_total, 2),
            "format_ast_success_rate": round(100.0 * self.format_ast_correct / self.format_total, 2),
        }
        for format_id, rows in sorted(self.by_format.items()):
            if rows:
                valid = sum(int(bool(row.get("format_valid"))) for row in rows)
                metrics[f"format_valid_rate/{format_id}"] = round(100.0 * valid / len(rows), 2)
        return metrics

    def get_metrics(self):
        metrics_dict = super().get_metrics()
        for _agg_mode, agg_metrics in metrics_dict.items():
            agg_metrics.update(self._format_metrics())
        return metrics_dict

    def metrics_to_print(self):
        metrics = super().metrics_to_print()
        if self.format_total > 0:
            metrics["format_valid_rate"] = as_percentage
            metrics["format_ast_success_rate"] = as_percentage
        return metrics


def _metrics_for_eval_mode(benchmark_metrics: dict[str, Any], eval_mode: str) -> dict[str, Any] | None:
    if eval_mode in benchmark_metrics:
        return benchmark_metrics[eval_mode]
    all_metrics = benchmark_metrics.get("_all_")
    if isinstance(all_metrics, dict):
        return all_metrics.get(eval_mode)
    return None


def compute_score(combined_metrics: dict) -> dict:
    """Aggregate NTT-COMPLEX subtest results."""
    benchmarks = {key: value for key, value in combined_metrics.items() if key == "ntt-complex.format_ast"}
    if not benchmarks:
        return {}

    eval_modes = set()
    for benchmark_metrics in benchmarks.values():
        if "_all_" in benchmark_metrics:
            eval_modes.update(benchmark_metrics["_all_"].keys())
        else:
            eval_modes.update(benchmark_metrics.keys())

    aggregated = {}
    for eval_mode in sorted(eval_modes):
        total_entries = 0
        weighted: dict[str, float] = defaultdict(float)
        for benchmark_metrics in benchmarks.values():
            metrics = _metrics_for_eval_mode(benchmark_metrics, eval_mode)
            if not metrics:
                continue
            entries = int(metrics.get("num_entries") or 0)
            if entries <= 0:
                continue
            total_entries += entries
            for key, value in metrics.items():
                if key == "num_entries" or not isinstance(value, (int, float)):
                    continue
                weighted[key] += float(value) * entries
        if total_entries <= 0:
            continue
        aggregated[eval_mode] = {"num_entries": total_entries}
        for key, value in weighted.items():
            aggregated[eval_mode][key] = round(value / total_entries, 2)
    return aggregated
