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


def compute_score(combined_metrics: dict) -> dict:
    """Aggregate metrics from the three ContextASR-Bench sub-benchmarks.

    Computes weighted averages of WER, NE-WER, NE-FNR across contextless,
    coarse, and fine evaluation modes.
    """
    main_names = ["contextless", "coarse", "fine"]
    benchmarks = {k: v for k, v in combined_metrics.items() if k.split(".")[-1] in main_names}

    if not benchmarks:
        return {}

    first_benchmark = next(iter(benchmarks.values()))
    eval_modes = list(first_benchmark.keys())

    aggregated = {}
    for eval_mode in eval_modes:
        total_entries = 0
        weighted_success = 0.0
        total_gen_seconds = 0
        weighted_tokens = 0.0
        weighted_wer = 0.0
        weighted_ne_wer = 0.0
        weighted_ne_fnr = 0.0
        wer_entries = 0
        ne_wer_entries = 0
        ne_fnr_entries = 0

        for benchmark_data in benchmarks.values():
            if eval_mode not in benchmark_data:
                continue

            metrics = benchmark_data[eval_mode]
            num_entries = metrics["num_entries"]
            if num_entries == 0:
                continue

            total_entries += num_entries
            weighted_success += metrics["success_rate"] * num_entries
            total_gen_seconds += metrics["gen_seconds"]
            weighted_tokens += metrics["avg_tokens"] * num_entries

            if "wer" in metrics:
                weighted_wer += metrics["wer"] * num_entries
                wer_entries += num_entries
            if "ne_wer" in metrics:
                weighted_ne_wer += metrics["ne_wer"] * num_entries
                ne_wer_entries += num_entries
            if "ne_fnr" in metrics:
                weighted_ne_fnr += metrics["ne_fnr"] * num_entries
                ne_fnr_entries += num_entries

        if total_entries == 0:
            continue

        agg = {
            "avg_tokens": int(weighted_tokens / total_entries),
            "gen_seconds": total_gen_seconds,
            "success_rate": weighted_success / total_entries,
            "num_entries": total_entries,
        }

        if wer_entries > 0:
            agg["wer"] = round(weighted_wer / wer_entries, 2)
        if ne_wer_entries > 0:
            agg["ne_wer"] = round(weighted_ne_wer / ne_wer_entries, 2)
        if ne_fnr_entries > 0:
            agg["ne_fnr"] = round(weighted_ne_fnr / ne_fnr_entries, 2)

        aggregated[eval_mode] = agg

    return aggregated
