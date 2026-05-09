# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
    """Aggregate metrics from audio benchmark groups with asr and st sub-benchmarks.

    Computes weighted averages (by num_entries) of per-task metrics; metrics
    that only one sub-benchmark emits (e.g. wer for asr, bleu for st) are
    averaged across the sub-benchmarks that have them.
    """
    main_names = ["asr", "st"]
    benchmarks = {k: v for k, v in combined_metrics.items() if k.split(".")[-1] in main_names}

    if not benchmarks:
        return {}

    weighted_metrics = ["wer", "wer_macro", "bleu", "comet"]
    summed_metrics = ["substitutions", "insertions", "deletions", "ref_words"]

    first_benchmark = next(iter(benchmarks.values()))
    eval_modes = list(first_benchmark.keys())

    aggregated = {}
    for eval_mode in eval_modes:
        total_entries = 0
        total_gen_seconds = 0
        weighted_success = 0.0
        weighted_tokens = 0.0
        weighted_no_answer = 0.0
        weighted_sums = {m: 0.0 for m in weighted_metrics}
        weighted_counts = {m: 0 for m in weighted_metrics}
        sums = {m: 0 for m in summed_metrics}

        for benchmark_data in benchmarks.values():
            if eval_mode not in benchmark_data:
                continue

            metrics = benchmark_data[eval_mode]
            num_entries = metrics.get("num_entries", 0)
            if num_entries == 0:
                continue

            total_entries += num_entries
            total_gen_seconds += metrics.get("gen_seconds", 0)
            weighted_success += metrics.get("success_rate", 0.0) * num_entries
            weighted_tokens += metrics.get("avg_tokens", 0.0) * num_entries
            weighted_no_answer += metrics.get("no_answer", 0.0) * num_entries

            for m in weighted_metrics:
                if m in metrics:
                    weighted_sums[m] += metrics[m] * num_entries
                    weighted_counts[m] += num_entries

            for m in summed_metrics:
                if m in metrics:
                    sums[m] += metrics[m]

        if total_entries == 0:
            continue

        agg = {
            "avg_tokens": int(weighted_tokens / total_entries),
            "gen_seconds": total_gen_seconds,
            "success_rate": weighted_success / total_entries,
            "no_answer": weighted_no_answer / total_entries,
            "num_entries": total_entries,
        }

        for m in weighted_metrics:
            if weighted_counts[m] > 0:
                agg[m] = round(weighted_sums[m] / weighted_counts[m], 2)

        for m in summed_metrics:
            if sums[m]:
                agg[m] = sums[m]

        aggregated[eval_mode] = agg

    return aggregated
