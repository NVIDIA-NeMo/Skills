# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


class AALCRMetrics(BaseMetrics):
    """Metrics for AA-LCR (Artificial Analysis Long Context Reading) dataset.

    This dataset uses an LLM-based equality checker (Officially, the non-reasoning Qwen3 235B A22B 2507)
    to evaluate whether candidate answers are consistent with official answers.
    """

    def __init__(self):
        super().__init__()
        # Track metrics by document category for detailed analysis
        self.category_metrics = defaultdict(lambda: defaultdict(float))
        self.category_totals = defaultdict(int)
        self.token_stats = defaultdict(list)  # Track input token statistics

        # Track accuracy vs token length buckets
        self.token_buckets = ["<80k", "80k-100k", "100k-110k", "110k-128k", "128k+"]
        self.token_bucket_metrics = defaultdict(
            lambda: defaultdict(lambda: {"correct": 0, "total": 0})
        )  # category -> bucket -> stats

    def reset(self):
        super().reset()
        self.category_metrics = defaultdict(lambda: defaultdict(float))
        self.category_totals = defaultdict(int)
        self.token_stats = defaultdict(list)
        self.token_bucket_metrics = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    @staticmethod
    def is_aalcr_correct(judgement: str) -> bool:
        """Check if AA-LCR judgement indicates correct answer.

        AA-LCR uses 'CORRECT' or 'INCORRECT' format instead of 'Judgement: Yes/No'.
        """
        if judgement is None:
            return False
        judgement = judgement.strip().upper()
        return judgement == "CORRECT" or judgement.startswith("CORRECT")

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        """Get correctness scores for a prediction using LLM-based equality checker."""
        correctness_dict = {}

        # Primary evaluation method: LLM-based equality checker
        if "judgement" in prediction:
            # Invalid generation: reasoning is not finished or non-reasoning generation is empty
            correctness_dict["generation_valid"] = len(prediction["generation"].strip()) > 0
            correctness_dict["judge_correct"] = (
                self.is_aalcr_correct(prediction["judgement"]) if correctness_dict["generation_valid"] else False
            )

        return correctness_dict

    def _get_token_bucket(self, input_tokens: int) -> str:
        """Get token bucket based on input token count."""
        if input_tokens < 80000:
            return "<80k"
        elif input_tokens < 100000:
            return "80k-100k"
        elif input_tokens < 110000:
            return "100k-110k"
        elif input_tokens < 128000:
            return "110k-128k"
        else:
            return "128k+"

    def _update_token_bucket_metrics(self, prediction: dict, score_dict: dict):
        """Update token bucket metrics for accuracy analysis."""
        category = prediction.get("document_category", "unknown")
        input_tokens = prediction.get("input_tokens")

        if input_tokens is not None:
            token_bucket = self._get_token_bucket(int(input_tokens))

            # Update total count for this category-bucket combination
            self.token_bucket_metrics[category][token_bucket]["total"] += 1

            # Update correct count based on judge_correct
            if score_dict.get("judge_correct", False):
                self.token_bucket_metrics[category][token_bucket]["correct"] += 1

    @classmethod
    def get_incorrect_sample(cls, prediction: dict) -> dict:
        """Return a prediction that evaluates as incorrect."""
        prediction = prediction.copy()
        prediction["judgement"] = "INCORRECT"
        prediction["predicted_answer"] = None
        return prediction

    def _update_category_metrics(self, prediction: dict, score_dict: dict):
        """Update per-category metrics if document_category is available."""
        category = prediction.get("document_category", "unknown")
        self.category_totals[category] += 1

        for score_method, is_correct in score_dict.items():
            if is_correct:
                self.category_metrics[category][score_method] += 1

    def _update_token_stats(self, prediction: dict):
        """Track input token statistics by category."""
        category = prediction.get("document_category", "unknown")
        input_tokens = prediction.get("input_tokens")
        if input_tokens is not None:
            self.token_stats[category].append(int(input_tokens))

    def update(self, predictions):
        """Update the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                Each prediction should contain 'judgement' from LLM equality checker.
        """
        super().update(predictions)

        # Update category metrics and token stats using the first prediction
        # (they should all have the same expected answer and metadata)
        if predictions:
            score_dict = self._get_score_dict(predictions[0])
            self._update_category_metrics(predictions[0], score_dict)
            self._update_token_stats(predictions[0])
            self._update_token_bucket_metrics(predictions[0], score_dict)

        # Compute standard pass@k and majority@k metrics
        # Here we use 'judgement' and 'generation' to calculate score and no_answer metric respectively
        predicted_answers = [pred.get("generation") for pred in predictions]

        self._compute_pass_at_k(predictions=predictions, predicted_answers=predicted_answers)
        self._compute_majority_at_k(predictions=predictions, predicted_answers=predicted_answers)

    def get_metrics(self):
        """Get all computed metrics including per-category breakdown and token statistics."""
        metrics_dict = super().get_metrics()

        # Add per-category metrics
        if self.category_totals:
            category_results = {}
            for category, total in self.category_totals.items():
                category_results[category] = {}
                for score_method, correct_count in self.category_metrics[category].items():
                    accuracy = 100.0 * correct_count / total
                    category_results[category][score_method] = accuracy

                    # Also add as top-level metric for table display
                    for eval_mode in metrics_dict:
                        metrics_dict[eval_mode][f"{category}_accuracy"] = accuracy

                category_results[category]["total_samples"] = total

                # Add token statistics
                if category in self.token_stats and self.token_stats[category]:
                    tokens = self.token_stats[category]
                    category_results[category]["avg_input_tokens"] = int(sum(tokens) / len(tokens))
                    category_results[category]["max_input_tokens"] = max(tokens)
                    category_results[category]["min_input_tokens"] = min(tokens)

            # Add category breakdown to the main evaluation mode
            for eval_mode in metrics_dict:
                if eval_mode == f"pass@1[avg-of-{self.max_k}]":  # Only add to the main evaluation mode
                    metrics_dict[eval_mode]["category_breakdown"] = category_results

        # Print category breakdown in a separate table
        if self.category_totals:
            self._print_category_table(category_results)

        # Print token length analysis
        self._print_token_length_analysis()

        return metrics_dict

    def _print_category_table(self, category_results):
        """Print category breakdown in a nicely formatted table."""
        if not category_results:
            return

        # Calculate column widths
        max_category_width = max(len(cat) for cat in category_results.keys())
        max_category_width = max(max_category_width, len("Category"))

        # Headers for the table
        headers = ["Category", "Samples", "Accuracy", "Avg Tokens", "Token Range"]
        col_widths = [max_category_width, 8, 10, 12, 15]

        total_width = sum(col_widths) + len(col_widths) * 3 - 3  # 3 chars per separator, minus last

        print("\n" + "=" * total_width)
        print(" AA-LCR Category Breakdown ".center(total_width))
        print("=" * total_width)

        # Print headers
        header_row = " | ".join([f"{headers[i]:<{col_widths[i]}}" for i in range(len(headers))])
        print(header_row)
        print("-" * total_width)

        # Print each category
        for category, stats in category_results.items():
            accuracy = stats.get("judge_correct", 0)
            samples = stats.get("total_samples", 0)
            avg_tokens = stats.get("avg_input_tokens", 0)
            min_tokens = stats.get("min_input_tokens", 0)
            max_tokens = stats.get("max_input_tokens", 0)

            token_range = f"{min_tokens}-{max_tokens}" if min_tokens and max_tokens else "N/A"

            row = [
                f"{category:<{col_widths[0]}}",
                f"{samples:<{col_widths[1]}}",
                f"{accuracy:.2f}%".ljust(col_widths[2]),
                f"{avg_tokens:<{col_widths[3]}}",
                f"{token_range:<{col_widths[4]}}",
            ]
            print(" | ".join(row))

        print("=" * total_width + "\n")

    def _print_token_length_analysis(self):
        """Print accuracy vs token length analysis across categories."""
        if not self.token_bucket_metrics:
            return

        # Calculate column widths
        categories = list(self.token_bucket_metrics.keys())
        if not categories:
            return

        max_category_width = max(len(cat) for cat in categories)
        max_category_width = max(max_category_width, len("Category"))

        col_widths = [max_category_width] + [12] * len(self.token_buckets)
        total_width = sum(col_widths) + len(col_widths) * 3 - 3

        print("=" * total_width)
        print(" Accuracy vs Token Length Analysis ".center(total_width))
        print("=" * total_width)

        # Print headers (split into two lines for better formatting)
        bucket_headers = " | ".join([f"{bucket:<{col_widths[i + 1]}}" for i, bucket in enumerate(self.token_buckets)])
        print(f"{'Category':<{col_widths[0]}} | {bucket_headers}")
        subheader = " | ".join([f"{'Acc% (#)':<{col_widths[i + 1]}}" for i in range(len(self.token_buckets))])
        print(f"{'':<{col_widths[0]}} | {subheader}")
        print("-" * total_width)

        # Print each category
        for category in categories:
            row_values = [f"{category:<{col_widths[0]}}"]

            for i, bucket in enumerate(self.token_buckets):
                bucket_stats = self.token_bucket_metrics[category][bucket]
                total = bucket_stats["total"]
                correct = bucket_stats["correct"]

                if total > 0:
                    accuracy = (correct / total) * 100
                    value = f"{accuracy:.1f}% ({total})"
                else:
                    value = "N/A (0)"

                row_values.append(f"{value:<{col_widths[i + 1]}}")

            print(" | ".join(row_values))

        # Print summary row with totals across categories
        print("-" * total_width)
        row_values = [f"{'OVERALL':<{col_widths[0]}}"]

        for i, bucket in enumerate(self.token_buckets):
            total_correct = 0
            total_samples = 0

            for category in categories:
                bucket_stats = self.token_bucket_metrics[category][bucket]
                total_correct += bucket_stats["correct"]
                total_samples += bucket_stats["total"]

            if total_samples > 0:
                overall_accuracy = (total_correct / total_samples) * 100
                value = f"{overall_accuracy:.1f}% ({total_samples})"
            else:
                value = "N/A (0)"

            row_values.append(f"{value:<{col_widths[i + 1]}}")

        print(" | ".join(row_values))
        print("=" * total_width + "\n")

    def evaluations_to_print(self):
        """Return which evaluations should be printed in the summary."""
        if self.max_k > 1:
            return [f"pass@1[avg-of-{self.max_k}]", f"majority@{self.max_k}", f"pass@{self.max_k}"]
        else:
            return ["pass@1"]

    def metrics_to_print(self):
        """Control which metrics are displayed in the summary table."""
        from nemo_skills.evaluation.metrics.base import default_formatting

        # Only show main metrics in the summary table since categories have their own table
        return {
            "judge_correct": default_formatting,
            "num_entries": default_formatting,
            "avg_tokens": default_formatting,
            "no_answer": default_formatting,
        }
