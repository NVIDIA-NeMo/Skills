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

"""Audio metrics: Metrics aggregation for audio evaluation tasks.

This module provides comprehensive metrics tracking and aggregation for various
audio-related evaluation tasks. It supports automatic metrics (WER, CER, BLEU, etc.)
as well as judge-based evaluation for open-ended audio tasks.

Supported Metrics:
- WER: Word Error Rate (standard ASR metric)
- WER_C: WER with capitalization
- WER_PC: WER with punctuation and capitalization
- PER: Punctuation Error Rate
- BLEU: Translation quality metric
- CER: Character Error Rate (for character-level evaluation)
- Hallucination Rate: Detection of hallucinated content
- PC Rate: Punctuation/Capitalization recovery rate

The metrics class is designed to be extensible, allowing easy addition of new
audio-specific metrics as the field evolves.
"""

import logging

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_float, as_int, as_percentage
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


class AudioMetrics(BaseMetrics):
    """Metrics class for audio evaluation tasks.

    This class tracks and aggregates various audio-specific metrics including
    error rates (WER, CER, PER), quality scores (BLEU), and advanced metrics
    like hallucination detection. It extends BaseMetrics to provide consistent
    metric computation and reporting across different audio tasks.
    """

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        """Initialize audio metrics with tracking lists for all supported metrics.

        Args:
            compute_no_answer: Whether to compute no_answer statistics
            max_k: Maximum k for pass@k and majority@k evaluation
        """
        super().__init__(compute_no_answer=compute_no_answer)
        self.max_k = max_k

        # Core audio metrics
        self.wer_scores = []
        self.wer_c_scores = []
        self.wer_pc_scores = []
        self.per_scores = []
        self.bleu_scores = []

        # Extended metrics
        self.cer_scores = []
        self.hallucination_scores = []
        self.pc_rate_scores = []
        self.punct_f1_scores = []
        self.cap_accuracy_scores = []
        self.total_hallucinated_chars = 0
        self.total_audio_seconds = 0.0

        # Judge scores (AudioBench-style rating 0-5, or legacy binary Yes/No mapped to 1/0)
        self.judge_ratings = []

    def _extract_judge_result(self, judgement_text: str) -> tuple[bool, float]:
        """Extract judge result from judgement text.

        Supports two formats:
        1. AudioBench format: 'Rating: X' where X is 0-5 (returns rating as float)
        2. Legacy/binary format: 'Judgement: Yes/No' (mapped to 5.0/0.0 for consistent 0-100 scaling)

        Returns:
            Tuple of (is_correct, rating_score)
            - is_correct: True if rating >= 3 (or Yes for legacy)
            - rating_score: 0-5 rating (or 0/5 for legacy binary)
        """
        import re

        # Try AudioBench format first: 'Rating: X'
        rating_match = re.search(r"Rating:\s*([0-9]+(?:\.[0-9]+)?)", judgement_text, re.IGNORECASE)
        if rating_match:
            rating = float(rating_match.group(1))
            rating = max(0.0, min(5.0, rating))
            return rating >= 3.0, rating

        # Try explicit Judgement: Yes/No format
        judgement_match = re.search(r"Judgement:\s*(Yes|No)", judgement_text, re.IGNORECASE)
        if judgement_match:
            is_yes = judgement_match.group(1).lower() == "yes"
            return is_yes, 5.0 if is_yes else 0.0

        # Last-resort: accept plain 'yes'/'no' anywhere in text
        if re.search(r"\byes\b", judgement_text, re.IGNORECASE):
            return True, 5.0
        if re.search(r"\bno\b", judgement_text, re.IGNORECASE):
            return False, 0.0

        return False, 0.0

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        """Extract correctness scores from prediction.

        Handles both automatic metrics and judge-based evaluation,
        determining the overall correctness of a prediction.

        Args:
            prediction: Prediction dictionary with metrics and/or judgement

        Returns:
            Dictionary with correctness scores
        """
        score_dict = {}

        category = prediction.get("category", "unknown")

        if "judgement" in prediction and category == "open":
            judge_correct, judge_rating = self._extract_judge_result(prediction["judgement"])
            score_dict["judge_correct"] = judge_correct
            score_dict["judge_rating"] = judge_rating

        if category == "open" and "judge_correct" in score_dict:
            score_dict["correct"] = score_dict["judge_correct"]
        elif "is_correct" in prediction:
            score_dict["correct"] = prediction["is_correct"]
        else:
            score_dict["correct"] = False

        return score_dict

    def get_incorrect_sample(self, prediction: dict) -> dict:
        """Return a sample marked as incorrect for all metrics.

        Used for handling error cases or missing predictions.

        Args:
            prediction: Prediction dictionary

        Returns:
            Updated prediction marked as incorrect
        """
        prediction = prediction.copy()
        prediction["is_correct"] = False
        prediction["judge_correct"] = False
        if not prediction.get("generation", "").strip():
            prediction["generation"] = None
        return prediction

    def update_common_metrics(self, agg_dict):
        """Override to always include avg_tokens even if 0 since it's in metrics_to_print.

        Updates common metrics like number of entries, average tokens, and generation time.

        Args:
            agg_dict: Dictionary to update with common metrics
        """
        agg_dict["num_entries"] = self.total
        agg_dict["avg_tokens"] = int(self.avg_tokens / self.total) if self.total > 0 else 0
        if self.max_end_time > float("-inf") and self.min_start_time < float("inf"):
            agg_dict["gen_seconds"] = int(self.max_end_time - self.min_start_time)

    def update(self, predictions):
        """Update metrics with new predictions.

        Collects all metric scores from predictions and updates internal tracking lists.
        Supports both existing metrics (WER, BLEU) and new metrics (CER, hallucination, PC rate).

        Args:
            predictions: List of prediction dictionaries with computed metrics
        """
        super().update(predictions)

        predicted_answers = [pred.get("generation", "").strip() or None for pred in predictions]

        # Collect existing metrics: WER, PnC, and BLEU scores
        for pred in predictions:
            if "wer" in pred and pred["wer"] is not None:
                self.wer_scores.append(pred["wer"])
            if "wer_c" in pred and pred["wer_c"] is not None:
                self.wer_c_scores.append(pred["wer_c"])
            if "wer_pc" in pred and pred["wer_pc"] is not None:
                self.wer_pc_scores.append(pred["wer_pc"])
            if "per" in pred and pred["per"] is not None:
                self.per_scores.append(pred["per"])
            if "bleu" in pred and pred["bleu"] is not None:
                self.bleu_scores.append(pred["bleu"])

            # Collect extended metrics
            if "cer" in pred and pred["cer"] is not None:
                self.cer_scores.append(pred["cer"])
            if "hallucination_rate" in pred and pred["hallucination_rate"] is not None:
                self.hallucination_scores.append(pred["hallucination_rate"])
            if "pc_rate" in pred and pred["pc_rate"] is not None:
                self.pc_rate_scores.append(pred["pc_rate"])
            if "punct_f1" in pred and pred["punct_f1"] is not None:
                self.punct_f1_scores.append(pred["punct_f1"])
            if "cap_accuracy" in pred and pred["cap_accuracy"] is not None:
                self.cap_accuracy_scores.append(pred["cap_accuracy"])

            if pred.get("task_type") == "Hallucination":
                predicted_text = pred.get("predicted_answer") or pred.get("generation") or ""
                audio_duration = pred.get("audio_duration", 0.0)
                if audio_duration > 0:
                    self.total_hallucinated_chars += len(predicted_text.strip())
                    self.total_audio_seconds += audio_duration

            # Collect judge ratings (0-5) from judge datasets if available
            score_dict = self._get_score_dict(pred)
            if "judge_rating" in score_dict:
                self.judge_ratings.append(score_dict["judge_rating"])

        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)

    def get_metrics(self):
        """Get computed metrics.

        Aggregates all collected metric scores and computes averages.
        Converts error rates and scores to percentages for reporting.

        Returns:
            Dictionary of aggregated metrics by evaluation mode
        """
        metrics_dict = super().get_metrics()

        for _agg_mode, agg_metrics in metrics_dict.items():
            if "no_answer" in agg_metrics:
                # Divide by 2.0 to compensate for double-counting in base metrics #MEL
                agg_metrics["no_answer"] = agg_metrics["no_answer"] / 2.0

            # Set success_rate based on correct field
            if "correct" in agg_metrics:
                agg_metrics["success_rate"] = agg_metrics["correct"]
            elif "judge_correct" in agg_metrics:
                agg_metrics["success_rate"] = agg_metrics["judge_correct"]

            # Add AudioBench-style judge_score if rating outputs were used.
            # Formula: judge_score = mean(ratings) * 20 (converts 0-5 scale to 0-100)
            if self.judge_ratings:
                avg_rating = sum(self.judge_ratings) / len(self.judge_ratings)
                agg_metrics["judge_score"] = avg_rating * 20

            # Add existing metrics: WER, PnC, and BLEU if available (convert to percentages and round to 2 decimals)
            if self.wer_scores:
                agg_metrics["wer"] = round(100.0 * sum(self.wer_scores) / len(self.wer_scores), 2)
            if self.wer_c_scores:
                agg_metrics["wer_c"] = round(100.0 * sum(self.wer_c_scores) / len(self.wer_c_scores), 2)
            if self.wer_pc_scores:
                agg_metrics["wer_pc"] = round(100.0 * sum(self.wer_pc_scores) / len(self.wer_pc_scores), 2)
            if self.per_scores:
                agg_metrics["per"] = round(100.0 * sum(self.per_scores) / len(self.per_scores), 2)
            if self.bleu_scores:
                agg_metrics["bleu"] = round(100.0 * sum(self.bleu_scores) / len(self.bleu_scores), 2)

            # Add extended metrics if available
            if self.cer_scores:
                agg_metrics["cer"] = round(100.0 * sum(self.cer_scores) / len(self.cer_scores), 2)
            if self.hallucination_scores:
                agg_metrics["hallucination_rate"] = round(
                    100.0 * sum(self.hallucination_scores) / len(self.hallucination_scores), 2
                )
            if self.pc_rate_scores:
                agg_metrics["pc_rate"] = round(100.0 * sum(self.pc_rate_scores) / len(self.pc_rate_scores), 2)
            if self.punct_f1_scores:
                agg_metrics["punct_f1"] = round(100.0 * sum(self.punct_f1_scores) / len(self.punct_f1_scores), 2)
            if self.cap_accuracy_scores:
                agg_metrics["cap_accuracy"] = round(
                    100.0 * sum(self.cap_accuracy_scores) / len(self.cap_accuracy_scores), 2
                )
            if self.total_audio_seconds > 0:
                total_minutes = self.total_audio_seconds / 60.0
                agg_metrics["char_rate"] = round(self.total_hallucinated_chars / total_minutes, 2)

        return metrics_dict

    def evaluations_to_print(self):
        """Specify which evaluation modes to print.

        Returns:
            List of evaluation mode names for display
        """
        evals = [f"pass@{self.max_k}"]
        if self.max_k > 1:
            evals.extend([f"majority@{self.max_k}", f"pass@1[avg-of-{self.max_k}]"])
        return evals

    def metrics_to_print(self):
        """Specify which metrics to print.

        Dynamically includes only the metrics that were actually computed
        based on the task types in the evaluation.

        Returns:
            Dictionary mapping metric names to formatting functions
        """
        base_metrics = {
            "avg_tokens": as_int,
            "gen_seconds": as_int,
            "success_rate": as_percentage,
        }

        if self.compute_no_answer:
            base_metrics["no_answer"] = as_percentage

        # AudioBench-style judge_score (0-100, not a percent)
        if self.judge_ratings:
            base_metrics["judge_score"] = lambda _k, v, _all: f"{v:.2f}"

        # Add existing metrics if they were computed
        if self.wer_scores:
            base_metrics["wer"] = as_percentage
        if self.wer_c_scores:
            base_metrics["wer_c"] = as_percentage
        if self.wer_pc_scores:
            base_metrics["wer_pc"] = as_percentage
        if self.per_scores:
            base_metrics["per"] = as_percentage
        if self.bleu_scores:
            base_metrics["bleu"] = as_percentage

        # Add extended metrics if they were computed
        if self.cer_scores:
            base_metrics["cer"] = as_percentage
        if self.hallucination_scores:
            base_metrics["hallucination_rate"] = as_percentage
        if self.pc_rate_scores:
            base_metrics["pc_rate"] = as_percentage
        if self.punct_f1_scores:
            base_metrics["punct_f1"] = as_percentage
        if self.cap_accuracy_scores:
            base_metrics["cap_accuracy"] = as_percentage
        if self.total_audio_seconds > 0:
            base_metrics["char_rate"] = as_float

        base_metrics["num_entries"] = as_int  # Add at end for better display order

        return base_metrics


def compute_score(combined_metrics: dict) -> dict:
    """
    Aggregate metrics from multiple sub-benchmarks into a single group score.

    This function is used for benchmark groups that contain multiple sub-benchmarks.
    It computes weighted averages across all sub-benchmarks based on the number of entries.

    Args:
        combined_metrics: Dictionary with benchmark names as keys.
                         Each benchmark has eval modes (e.g., 'pass@1') as keys,
                         which contain the actual metrics.
                         Format: {benchmark_name: {eval_mode: {metrics...}}}

    Returns:
        Aggregated metrics dictionary in the same format, with weighted averages
        computed across all sub-benchmarks.
    """
    # Identify main benchmark categories (nonjudge, judge)
    main_benchmark_names = ["nonjudge", "judge"]
    benchmarks = {k: v for k, v in combined_metrics.items() if k.split(".")[-1] in main_benchmark_names}

    if not benchmarks:
        return {}

    # Get all eval modes from first benchmark (they should all have the same modes)
    first_benchmark = next(iter(benchmarks.values()))
    eval_modes = list(first_benchmark.keys())

    # Aggregate metrics for each evaluation mode
    aggregated = {}
    for eval_mode in eval_modes:
        total_entries = 0
        weighted_success = 0.0
        total_gen_seconds = 0
        weighted_tokens = 0.0
        weighted_no_answer = 0.0

        for benchmark_name, benchmark_data in benchmarks.items():
            if eval_mode not in benchmark_data:
                continue

            metrics = benchmark_data[eval_mode]
            num_entries = metrics.get("num_entries", 0)
            total_entries += num_entries

            # Aggregate weighted by number of entries (metrics are already percentages)
            if num_entries > 0:
                weighted_success += metrics.get("success_rate", 0.0) * num_entries
                total_gen_seconds += metrics.get("gen_seconds", 0)
                weighted_tokens += metrics.get("avg_tokens", 0.0) * num_entries
                weighted_no_answer += metrics.get("no_answer", 0.0) * num_entries

        # Compute aggregated metrics
        aggregated[eval_mode] = {
            "avg_tokens": int(weighted_tokens / total_entries) if total_entries > 0 else 0,
            "gen_seconds": total_gen_seconds,
            "success_rate": weighted_success / total_entries if total_entries > 0 else 0.0,
            "no_answer": weighted_no_answer / total_entries if total_entries > 0 else 0.0,
            "num_entries": total_entries,
        }

    return aggregated
