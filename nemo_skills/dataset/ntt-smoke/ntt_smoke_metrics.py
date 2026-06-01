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

"""Metrics and group scoring for NTT-SMOKE."""

from __future__ import annotations

from collections import defaultdict
from math import sqrt
from statistics import mean
from typing import Any

from nemo_skills.evaluation.metrics.audio_metrics import AudioMetrics
from nemo_skills.evaluation.metrics.base import as_float, as_int, as_percentage


SUCCESS_WER_THRESHOLD = 0.05


class NTTSmokeMetrics(AudioMetrics):
    """AudioMetrics extension with NTT-SMOKE subtest signals."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer, max_k=max_k)
        self.strict_hallucinations = 0.0
        self.strict_hallucination_total = 0
        self.total_nonempty_chars = 0

        self.ne_wer_total_errors = 0
        self.ne_wer_total_ref_words = 0
        self.ne_fnr_total_hits = 0
        self.ne_fnr_total_entities = 0

        self.prompt_group_wers: dict[str, list[float]] = defaultdict(list)
        self.prompt_group_texts: dict[str, set[str]] = defaultdict(set)
        self.language_wers: dict[str, list[float]] = defaultdict(list)
        self.strict_hallucination_scores: list[float] = []
        self.correct_scores: list[float] = []
        self.success_wer_thresholds: list[float] = []

    @staticmethod
    def _ci95(values: list[float], scale: float = 100.0) -> float | None:
        """Return a normal-approximation 95% CI half-width."""
        if len(values) < 2:
            return None
        avg = mean(values)
        variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
        return round(1.96 * sqrt(variance / len(values)) * scale, 2)

    @staticmethod
    def _score_missing_fields(pred: dict[str, Any]) -> dict[str, Any]:
        """Compute NTT row metrics for generation-only baseline outputs."""
        if pred.get("wer") is not None or pred.get("is_correct") is not None:
            return pred
        if "generation" not in pred:
            return pred

        from importlib import import_module

        evaluator = import_module("nemo_skills.dataset.ntt-smoke.ntt_smoke_eval")
        audio_config = evaluator._audio_config({})
        task_type = pred.get("task_type")
        generation = str(pred.get("generation") or "")

        if task_type == "ContextASR":
            updates = evaluator._evaluate_contextasr_sample(
                pred,
                evaluator._clean_generation(generation, audio_config),
                SUCCESS_WER_THRESHOLD,
            )
        elif task_type == "Text-MCQ":
            updates = evaluator._evaluate_text_mcq(pred)
        elif task_type == "PreferenceASR":
            updates = evaluator._evaluate_preference_asr_sample(
                pred,
                generation,
                audio_config,
                SUCCESS_WER_THRESHOLD,
            )
        else:
            updates = evaluator._add_wer_correct_words(evaluator.evaluate_audio_sample(pred, audio_config))
            evaluator._apply_wer_success_threshold(updates, SUCCESS_WER_THRESHOLD)
            if task_type == "Hallucination":
                evaluator._add_strict_hallucination(updates, evaluator._clean_generation(generation, audio_config))

        pred.update(updates)
        return pred

    def update(self, predictions):
        thresholded_predictions = []
        for pred in predictions:
            pred = self._score_missing_fields(dict(pred))
            if pred.get("wer") is not None:
                threshold = float(pred.get("success_wer_threshold", SUCCESS_WER_THRESHOLD))
                pred["success_wer_threshold"] = threshold
                pred["success_wer_threshold_percent"] = round(100.0 * threshold, 6)
                pred["is_correct"] = float(pred["wer"]) < threshold
                self.success_wer_thresholds.append(threshold)

            base_pred = dict(pred)
            base_pred.pop("wer_correct_words", None)
            thresholded_predictions.append(base_pred)

            if "strict_hallucination_rate" in pred:
                strict_score = float(pred["strict_hallucination_rate"])
                self.strict_hallucinations += strict_score
                self.strict_hallucination_total += 1
                self.total_nonempty_chars += int(pred.get("nonempty_chars") or 0)
                self.strict_hallucination_scores.append(strict_score)

            if pred.get("is_correct") is not None:
                self.correct_scores.append(1.0 if pred.get("is_correct") else 0.0)

            if "ne_wer_errors" in pred and "ne_wer_ref_words" in pred:
                self.ne_wer_total_errors += int(pred["ne_wer_errors"])
                self.ne_wer_total_ref_words += int(pred["ne_wer_ref_words"])
                self.ne_fnr_total_hits += int(pred.get("ne_fnr_hits") or 0)
                self.ne_fnr_total_entities += int(pred.get("ne_fnr_total") or 0)

            prompt_group_id = pred.get("prompt_group_id")
            if prompt_group_id and pred.get("wer") is not None:
                self.prompt_group_wers[str(prompt_group_id)].append(float(pred["wer"]))
                self.prompt_group_texts[str(prompt_group_id)].add(str(pred.get("pred_text") or "").strip())

            language = pred.get("language")
            if not language:
                extra_fields = pred.get("extra_fields") or {}
                language = extra_fields.get("src_lang")
            if language and pred.get("wer") is not None:
                self.language_wers[str(language)].append(float(pred["wer"]))

        super().update(thresholded_predictions)

    def _prompt_metrics(self) -> dict[str, float | int]:
        groups = [values for values in self.prompt_group_wers.values() if len(values) > 1]
        if not groups:
            return {}

        deltas = [max(values) - min(values) for values in groups]
        text_match = [
            1.0 if len(self.prompt_group_texts[group_id]) <= 1 else 0.0
            for group_id, values in self.prompt_group_wers.items()
            if len(values) > 1
        ]
        return {
            "prompt_groups": len(groups),
            "prompt_wer_delta": round(100.0 * mean(deltas), 2),
            "prompt_text_match_rate": round(100.0 * mean(text_match), 2),
            "prompt_wer_delta_ci95": self._ci95(deltas) or 0.0,
            "prompt_text_match_rate_ci95": self._ci95(text_match) or 0.0,
        }

    def _language_metrics(self) -> dict[str, float | int]:
        if not self.language_wers:
            return {}
        per_language = [100.0 * mean(values) for values in self.language_wers.values() if values]
        if not per_language:
            return {}
        return {
            "language_count": len(per_language),
            "language_wer_macro": round(mean(per_language), 2),
            "language_wer_macro_ci95": self._ci95([value / 100.0 for value in per_language]) or 0.0,
        }

    def _success_threshold_metrics(self) -> dict[str, float | int]:
        if not self.success_wer_thresholds:
            return {}
        unique_thresholds = sorted({round(value, 10) for value in self.success_wer_thresholds})
        metrics = {
            "success_wer_threshold": unique_thresholds[0],
            "success_wer_threshold_percent": round(100.0 * unique_thresholds[0], 6),
        }
        if len(unique_thresholds) > 1:
            metrics["success_wer_threshold_mixed"] = 1
        return metrics

    def get_metrics(self):
        metrics_dict = super().get_metrics()

        for _agg_mode, agg_metrics in metrics_dict.items():
            agg_metrics.update(self._success_threshold_metrics())

            if (
                "ref_words" in agg_metrics
                and "substitutions" in agg_metrics
                and "deletions" in agg_metrics
                and "correct_words" not in agg_metrics
            ):
                agg_metrics["correct_words"] = max(
                    0,
                    int(agg_metrics["ref_words"]) - int(agg_metrics["substitutions"]) - int(agg_metrics["deletions"]),
                )

            if self.strict_hallucination_total > 0:
                agg_metrics["strict_hallucination_rate"] = round(
                    100.0 * self.strict_hallucinations / self.strict_hallucination_total, 2
                )
                strict_ci95 = self._ci95(self.strict_hallucination_scores)
                if strict_ci95 is not None:
                    agg_metrics["strict_hallucination_rate_ci95"] = strict_ci95
                agg_metrics["avg_nonempty_chars"] = round(
                    self.total_nonempty_chars / self.strict_hallucination_total, 2
                )

            if self.ne_wer_total_ref_words > 0:
                agg_metrics["ne_wer"] = round(100.0 * self.ne_wer_total_errors / self.ne_wer_total_ref_words, 2)
            if self.ne_fnr_total_entities > 0:
                agg_metrics["ne_fnr"] = round(
                    100.0 * (1.0 - self.ne_fnr_total_hits / self.ne_fnr_total_entities), 2
                )

            agg_metrics.update(self._prompt_metrics())
            agg_metrics.update(self._language_metrics())
            if self.wer_scores:
                wer_macro_ci95 = self._ci95([float(value) for value in self.wer_scores])
                if wer_macro_ci95 is not None:
                    agg_metrics["wer_macro_ci95"] = wer_macro_ci95
            if self.hallucination_scores:
                hallucination_ci95 = self._ci95([float(value) for value in self.hallucination_scores])
                if hallucination_ci95 is not None:
                    agg_metrics["hallucination_rate_ci95"] = hallucination_ci95
            if self.correct_scores:
                success_ci95 = self._ci95(self.correct_scores)
                if success_ci95 is not None:
                    agg_metrics["success_rate_ci95"] = success_ci95

        return metrics_dict

    def metrics_to_print(self):
        metrics = super().metrics_to_print()
        if self.strict_hallucination_total > 0:
            metrics["strict_hallucination_rate"] = as_percentage
            metrics["strict_hallucination_rate_ci95"] = as_percentage
            metrics["avg_nonempty_chars"] = as_float
        if self.ne_wer_total_ref_words > 0:
            metrics["ne_wer"] = as_percentage
        if self.ne_fnr_total_entities > 0:
            metrics["ne_fnr"] = as_percentage
        if self.prompt_group_wers:
            metrics["prompt_groups"] = as_int
            metrics["prompt_wer_delta"] = as_percentage
            metrics["prompt_wer_delta_ci95"] = as_percentage
            metrics["prompt_text_match_rate"] = as_percentage
            metrics["prompt_text_match_rate_ci95"] = as_percentage
        if self.language_wers:
            metrics["language_count"] = as_int
            metrics["language_wer_macro"] = as_percentage
            metrics["language_wer_macro_ci95"] = as_percentage
        if self.wer_total_ref_words > 0:
            metrics["correct_words"] = as_int
        if self.wer_scores:
            metrics["wer_macro_ci95"] = as_percentage
        if self.hallucination_scores:
            metrics["hallucination_rate_ci95"] = as_percentage
        if self.correct_scores:
            metrics["success_rate_ci95"] = as_percentage
        if self.success_wer_thresholds:
            metrics["success_wer_threshold"] = as_float
            metrics["success_wer_threshold_percent"] = as_percentage
            if len({round(value, 10) for value in self.success_wer_thresholds}) > 1:
                metrics["success_wer_threshold_mixed"] = as_int
        return metrics


def _metrics_for_eval_mode(benchmark_metrics: dict[str, Any], eval_mode: str) -> dict[str, Any] | None:
    if eval_mode in benchmark_metrics:
        return benchmark_metrics[eval_mode]
    all_metrics = benchmark_metrics.get("_all_")
    if isinstance(all_metrics, dict):
        return all_metrics.get(eval_mode)
    return None


def compute_score(combined_metrics: dict) -> dict:
    """Aggregate NTT-SMOKE English group results."""
    benchmarks = {key: value for key, value in combined_metrics.items() if key == "ntt-smoke.en"}
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
        sum_like = {
            "gen_seconds",
            "substitutions",
            "insertions",
            "deletions",
            "ref_words",
            "correct_words",
            "prompt_groups",
            "language_count",
        }
        config_like = {"success_wer_threshold", "success_wer_threshold_percent", "success_wer_threshold_mixed"}
        integer_like = {"avg_tokens", *sum_like}

        for benchmark_metrics in benchmarks.values():
            metrics = _metrics_for_eval_mode(benchmark_metrics, eval_mode)
            if not metrics:
                continue
            entries = int(metrics.get("num_entries") or 0)
            if entries <= 0:
                continue
            total_entries += entries
            for key, value in metrics.items():
                if (
                    key == "num_entries"
                    or key.endswith("_ci95")
                    or key in config_like
                    or not isinstance(value, (int, float))
                ):
                    continue
                weight = 1 if key in sum_like else entries
                weighted[key] += float(value) * weight

        config_values: dict[str, set[float]] = defaultdict(set)
        for benchmark_metrics in benchmarks.values():
            metrics = _metrics_for_eval_mode(benchmark_metrics, eval_mode)
            if not metrics:
                continue
            for key in config_like:
                value = metrics.get(key)
                if isinstance(value, (int, float)):
                    config_values[key].add(round(float(value), 10))

        if total_entries <= 0:
            continue

        mode_metrics = {"num_entries": total_entries}
        for key, value in weighted.items():
            if key in sum_like:
                mode_metrics[key] = int(value)
            elif key == "avg_tokens":
                mode_metrics[key] = int(value / total_entries)
            elif key in integer_like:
                mode_metrics[key] = int(value)
            else:
                mode_metrics[key] = round(value / total_entries, 2)
        for key, values in config_values.items():
            if values:
                value = sorted(values)[0]
                mode_metrics[key] = int(value) if key.endswith("_mixed") else value
        aggregated[eval_mode] = mode_metrics

    return aggregated
