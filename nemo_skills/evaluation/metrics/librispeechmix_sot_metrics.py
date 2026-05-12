# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Metrics aggregation for LibriSpeechMix SOT cpWER."""

from __future__ import annotations

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_float, as_int, as_percentage


class LibriSpeechMixSOTMetrics(BaseMetrics):
    """Aggregate LibriSpeechMix SOT cpWER from raw corpus counts."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer)
        self.max_k = max_k

    def _get_score_dict(self, prediction: dict) -> dict[str, bool]:
        return {"correct": bool(prediction.get("is_correct", False))}

    def get_incorrect_sample(self, prediction: dict) -> dict:
        prediction = prediction.copy()
        prediction["is_correct"] = False
        prediction["cpwer_errors"] = prediction.get("cpwer_ref_words", 0)
        prediction["cpwer_deletions"] = prediction.get("cpwer_ref_words", 0)
        prediction["cpwer_insertions"] = 0
        prediction["cpwer_substitutions"] = 0
        return prediction

    def update(self, predictions: list[dict]) -> None:
        super().update(predictions)
        predicted_answers = [p.get("generation") or p.get("pred_text") or None for p in predictions]
        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        if len(predictions) > 1:
            self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)

        first = predictions[0]
        for mode in ["pass@1", "pass@1[avg-of-1]"]:
            self.eval_dict[mode]["cpwer_errors"] += first.get("cpwer_errors", 0)
            self.eval_dict[mode]["cpwer_substitutions"] += first.get("cpwer_substitutions", 0)
            self.eval_dict[mode]["cpwer_insertions"] += first.get("cpwer_insertions", 0)
            self.eval_dict[mode]["cpwer_deletions"] += first.get("cpwer_deletions", 0)
            self.eval_dict[mode]["cpwer_ref_words"] += first.get("cpwer_ref_words", 0)

    def get_metrics(self) -> dict:
        metrics_dict = super().get_metrics()
        raw_count_keys = [
            "cpwer_errors",
            "cpwer_substitutions",
            "cpwer_insertions",
            "cpwer_deletions",
            "cpwer_ref_words",
        ]
        for mode, metrics in metrics_dict.items():
            raw_metrics = self.eval_dict.get(mode, {})
            for key in raw_count_keys:
                metrics[key] = int(raw_metrics.get(key, 0))
            ref_words = metrics.get("cpwer_ref_words", 0)
            errors = metrics.get("cpwer_errors", 0)
            if ref_words > 0:
                metrics["cpwer"] = round(100.0 * errors / ref_words, 2)
        return metrics_dict

    def evaluations_to_print(self) -> list[str]:
        return [f"pass@{self.max_k}"]

    def metrics_to_print(self) -> dict:
        metrics = {
            "correct": as_percentage,
            "cpwer": as_float,
            "cpwer_errors": as_int,
            "cpwer_substitutions": as_int,
            "cpwer_insertions": as_int,
            "cpwer_deletions": as_int,
            "cpwer_ref_words": as_int,
            "num_entries": as_int,
        }
        if self.compute_no_answer:
            metrics["no_answer"] = as_percentage
        return metrics
