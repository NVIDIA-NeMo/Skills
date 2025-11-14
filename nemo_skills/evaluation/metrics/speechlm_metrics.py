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

import logging

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_int, as_percentage
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


class SpeechLMMetrics(BaseMetrics):
    """Metrics class for speech/audio language model evaluation tasks."""

    def __init__(self, compute_no_answer: bool = True, max_k: int = 1):
        super().__init__(compute_no_answer=compute_no_answer)
        self.max_k = max_k
        self.wer_scores = []
        self.bleu_scores = []

    def _extract_judge_result(self, judgement_text: str) -> bool:
        """Extract judge result from judgement text."""
        import re

        if re.search(r"\byes\b", judgement_text, re.IGNORECASE):
            return True
        elif re.search(r"\bno\b", judgement_text, re.IGNORECASE):
            return False
        else:
            return False

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        """Extract correctness scores from prediction."""
        score_dict = {}

        category = prediction.get("category", "unknown")

        if "judgement" in prediction and category == "open":
            judge_result = self._extract_judge_result(prediction["judgement"])
            score_dict["judge_correct"] = judge_result

        if category == "open" and "judge_correct" in score_dict:
            score_dict["correct"] = score_dict["judge_correct"]
        elif "is_correct" in prediction:
            score_dict["correct"] = prediction["is_correct"]
        else:
            score_dict["correct"] = False

        return score_dict

    def get_incorrect_sample(self, prediction: dict) -> dict:
        """Return a sample marked as incorrect for all metrics."""
        prediction = prediction.copy()
        prediction["is_correct"] = False
        prediction["judge_correct"] = False
        if not prediction.get("generation", "").strip():
            prediction["generation"] = None
        return prediction

    def update_common_metrics(self, agg_dict):
        """Override to always include avg_tokens even if 0 since it's in metrics_to_print."""
        agg_dict["num_entries"] = self.total
        agg_dict["avg_tokens"] = int(self.avg_tokens / self.total) if self.total > 0 else 0
        if self.max_end_time > float("-inf") and self.min_start_time < float("inf"):
            agg_dict["gen_seconds"] = int(self.max_end_time - self.min_start_time)

    def update(self, predictions):
        """Update metrics with new predictions."""
        super().update(predictions)

        predicted_answers = [pred.get("generation", "").strip() or None for pred in predictions]

        # Collect WER and BLEU scores
        for pred in predictions:
            if "wer" in pred and pred["wer"] is not None:
                self.wer_scores.append(pred["wer"])
            if "bleu" in pred and pred["bleu"] is not None:
                self.bleu_scores.append(pred["bleu"])

        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)

    def get_metrics(self):
        """Get computed metrics."""
        metrics_dict = super().get_metrics()

        for agg_mode, agg_metrics in metrics_dict.items():
            if "no_answer" in agg_metrics:
                agg_metrics["no_answer"] = agg_metrics["no_answer"] / 2.0

            # Set success_rate based on correct field
            if "correct" in agg_metrics:
                agg_metrics["success_rate"] = agg_metrics["correct"]
            elif "judge_correct" in agg_metrics:
                agg_metrics["success_rate"] = agg_metrics["judge_correct"]
            
            # Add WER and BLEU if available
            if self.wer_scores:
                agg_metrics["wer"] = sum(self.wer_scores) / len(self.wer_scores)
            if self.bleu_scores:
                agg_metrics["bleu"] = sum(self.bleu_scores) / len(self.bleu_scores)

        return metrics_dict

    def evaluations_to_print(self):
        """Specify which evaluation modes to print."""
        evals = [f"pass@{self.max_k}"]
        if self.max_k > 1:
            evals.extend([f"majority@{self.max_k}", f"pass@1[avg-of-{self.max_k}]"])
        return evals

    def metrics_to_print(self):
        """Specify which metrics to print."""
        base_metrics = {
            "avg_tokens": as_int,
            "gen_seconds": as_int,
            "success_rate": as_percentage,
        }

        if self.compute_no_answer:
            base_metrics["no_answer"] = as_percentage
        
        # Add WER and BLEU if they were computed
        if self.wer_scores:
            base_metrics["wer"] = as_percentage
        if self.bleu_scores:
            base_metrics["bleu"] = as_percentage

        base_metrics["num_entries"] = as_int  # Add at end for better display order

        return base_metrics


def compute_score(combined_metrics: dict) -> dict:
    """
    Aggregate metrics from multiple sub-benchmarks into a single group score.

    Args:
        combined_metrics: Dictionary with benchmark names as keys.
                         Each benchmark has eval modes (e.g., 'pass@1') as keys,
                         which contain the actual metrics.
                         Format: {benchmark_name: {eval_mode: {metrics...}}}

    Returns:
        Aggregated metrics dictionary in the same format.
    """
    # Identify main benchmark categories (nonjudge, judge)
    main_benchmark_names = ['nonjudge', 'judge']
    benchmarks = {k: v for k, v in combined_metrics.items()
                  if k.split('.')[-1] in main_benchmark_names}

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
            num_entries = metrics.get('num_entries', 0)
            total_entries += num_entries

            # Aggregate weighted by number of entries (metrics are already percentages)
            if num_entries > 0:
                weighted_success += metrics.get('success_rate', 0.0) * num_entries
                total_gen_seconds += metrics.get('gen_seconds', 0)
                weighted_tokens += metrics.get('avg_tokens', 0.0) * num_entries
                weighted_no_answer += metrics.get('no_answer', 0.0) * num_entries

        # Compute aggregated metrics
        aggregated[eval_mode] = {
            'avg_tokens': int(weighted_tokens / total_entries) if total_entries > 0 else 0,
            'gen_seconds': total_gen_seconds,
            'success_rate': weighted_success / total_entries if total_entries > 0 else 0.0,
            'no_answer': weighted_no_answer / total_entries if total_entries > 0 else 0.0,
            'num_entries': total_entries,
        }

    return aggregated
