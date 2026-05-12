# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Group score aggregation for LibriSpeechMix SOT."""

from __future__ import annotations


def _aggregate(benchmarks: dict, eval_mode: str) -> dict:
    total_entries = 0
    errors = 0
    substitutions = 0
    insertions = 0
    deletions = 0
    ref_words = 0

    for benchmark_metrics in benchmarks.values():
        metrics = benchmark_metrics.get(eval_mode)
        if not metrics:
            continue
        total_entries += metrics.get("num_entries", 0)
        errors += metrics.get("cpwer_errors", 0)
        substitutions += metrics.get("cpwer_substitutions", 0)
        insertions += metrics.get("cpwer_insertions", 0)
        deletions += metrics.get("cpwer_deletions", 0)
        ref_words += metrics.get("cpwer_ref_words", 0)

    if total_entries == 0:
        return {}

    output = {
        "num_entries": total_entries,
        "cpwer_errors": errors,
        "cpwer_substitutions": substitutions,
        "cpwer_insertions": insertions,
        "cpwer_deletions": deletions,
        "cpwer_ref_words": ref_words,
    }
    if ref_words > 0:
        output["cpwer"] = round(100.0 * errors / ref_words, 2)
    return output


def compute_score(combined_metrics: dict) -> dict:
    """Aggregate cpWER overall and by duration split."""
    if not combined_metrics:
        return {}

    first_benchmark = next(iter(combined_metrics.values()))
    eval_modes = list(first_benchmark.keys())
    groups = {
        "all": combined_metrics,
        "under20s": {k: v for k, v in combined_metrics.items() if ".under20s-" in k},
        "over20s": {k: v for k, v in combined_metrics.items() if ".over20s-" in k},
        "test_clean": {k: v for k, v in combined_metrics.items() if "-test-clean-" in k},
        "dev_clean": {k: v for k, v in combined_metrics.items() if "-dev-clean-" in k},
    }

    output = {}
    for eval_mode in eval_modes:
        mode_output = {}
        for group_name, group_metrics in groups.items():
            aggregate = _aggregate(group_metrics, eval_mode)
            if aggregate:
                mode_output[group_name] = aggregate
        if mode_output:
            output[eval_mode] = mode_output
    return output
