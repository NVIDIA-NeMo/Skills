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

"""Group-level aggregation for the CS-FLEURS code-switched ASR benchmark.

Each sub-benchmark (read / mms / xtts-test1 / xtts-test2) already reports a
per-language-pair WER/CER breakdown via ``subset_for_metrics``. This module adds
a per-test-set headline number (entry-weighted WER/CER, where CER is stored
under the ``wer`` key for consistency with corpus-level aggregation) plus an
overall entry-weighted figure across the prepared test sets.

Note: the test sets differ in nature — ``read`` is human-validated while the
others are synthetic (XTTS / MMS) — so the per-test-set numbers are the
meaningful comparison; the overall figure is a convenience aggregate only.
"""

SUBSET_NAMES = ["read", "mms", "xtts-test1", "xtts-test2"]


def compute_score(combined_metrics: dict) -> dict:
    """Aggregate CS-FLEURS sub-benchmark metrics into per-test-set + overall scores.

    ``combined_metrics`` maps each sub-benchmark name (e.g. ``cs-fleurs.read``)
    to a dict of eval-mode -> metrics dict (as emitted by the audio metrics
    computation). Returns ``{eval_mode: {subset_name: {...}, "overall": {...}}}``.
    """
    benchmarks = {k: v for k, v in combined_metrics.items() if k.split(".")[-1] in SUBSET_NAMES}
    if not benchmarks:
        return {}

    weighted_metrics = ["wer", "wer_macro"]
    summed_metrics = ["substitutions", "insertions", "deletions", "ref_words"]

    first_benchmark = next(iter(benchmarks.values()))
    eval_modes = list(first_benchmark.keys())

    def _summarize(metrics_list: list[dict]) -> dict | None:
        total_entries = 0
        total_gen_seconds = 0
        weighted_success = 0.0
        weighted_tokens = 0.0
        weighted_no_answer = 0.0
        weighted_sums = {m: 0.0 for m in weighted_metrics}
        weighted_counts = {m: 0 for m in weighted_metrics}
        sums = {m: 0 for m in summed_metrics}

        for metrics in metrics_list:
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
            return None

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
        return agg

    aggregated: dict[str, dict] = {}
    for eval_mode in eval_modes:
        per_subset = {}
        for name, benchmark_data in benchmarks.items():
            if eval_mode not in benchmark_data:
                continue
            summary = _summarize([benchmark_data[eval_mode]])
            if summary is not None:
                per_subset[name.split(".")[-1]] = summary

        if not per_subset:
            continue

        overall = _summarize(list(per_subset.values()))
        if overall is not None:
            per_subset["overall"] = overall
        aggregated[eval_mode] = per_subset

    return aggregated
