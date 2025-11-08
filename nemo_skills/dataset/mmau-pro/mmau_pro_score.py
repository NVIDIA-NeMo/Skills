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


def compute_score(combined_metrics: dict) -> dict:
    """Aggregate metrics from multiple sub-benchmarks into a single group score."""

    # Only include the three MAIN benchmarks, not subcategories
    main_benchmark_names = ["closed_form", "open_ended", "instruction_following"]
    mmau_benchmarks = {k: v for k, v in combined_metrics.items() if k.split(".")[-1] in main_benchmark_names}

    if not mmau_benchmarks:
        return {}

    # Get all eval modes from first benchmark (they should all have the same modes)
    first_benchmark = next(iter(mmau_benchmarks.values()))
    eval_modes = list(first_benchmark.keys())

    # Aggregate metrics for each evaluation mode
    aggregated = {}
    for eval_mode in eval_modes:
        total_entries = 0
        weighted_success = 0.0
        total_gen_seconds = 0
        weighted_tokens = 0.0
        weighted_no_answer = 0.0

        for benchmark_name, benchmark_data in mmau_benchmarks.items():
            if eval_mode not in benchmark_data:
                continue

            metrics = benchmark_data[eval_mode]
            num_entries = metrics["num_entries"]
            total_entries += num_entries

            # Aggregate weighted by number of entries (metrics are already percentages)
            weighted_success += metrics["success_rate"] * num_entries
            total_gen_seconds += metrics["gen_seconds"]
            weighted_tokens += metrics["avg_tokens"] * num_entries
            weighted_no_answer += metrics["no_answer"] * num_entries

        # Compute aggregated metrics
        aggregated[eval_mode] = {
            "avg_tokens": int(weighted_tokens / total_entries) if total_entries > 0 else 0,
            "gen_seconds": total_gen_seconds,
            "success_rate": weighted_success / total_entries if total_entries > 0 else 0.0,
            "no_answer": weighted_no_answer / total_entries if total_entries > 0 else 0.0,
            "num_entries": total_entries,
        }

    return aggregated
