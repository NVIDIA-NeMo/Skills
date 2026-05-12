#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""
Compute submission counts per problem from cluster files with scores.

Rules per problem file (<p>_cluster.jsonl):
1) If no solution in any cluster has score == True -> submission_count = -1
2) Otherwise:
   - Sort clusters by cluster["tournament_wins"] (desc; missing -> 0)
   - Round-robin over clusters:
       First pass: try largest-token solution from each cluster in order
       Second pass: try 2nd-largest token solution from each cluster in order
       ...
     Stop when we encounter a solution with score == True.
     submission_count = number of attempts made up to and including the success.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


def to_bool(value: Any) -> bool:
    """Convert common truthy/falsey values to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def to_int(value: Any, default: int = 0) -> int:
    """Convert value to int, returning default on failure."""
    try:
        return int(value)
    except Exception:
        return default


def to_float(value: Any, default: float = 0.0) -> float:
    """Convert value to float, returning default on failure."""
    try:
        return float(value)
    except Exception:
        return default


def load_clusters(path: Path) -> Dict[str, Any]:
    """Load a cluster JSON payload and return empty dict on failure."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"    Skip {path.name}: {e}")
        return {}


def extract_problem_number(filename: str) -> int:
    """Extract numeric problem id from '<id>_cluster.jsonl' filename."""
    try:
        return int(os.path.basename(filename).split("_", 1)[0])
    except Exception:
        return 10**9


def any_solution_true(clusters_payload: Dict[str, Any]) -> bool:
    """Check whether any solution in the payload has a true score."""
    for val in clusters_payload.values():
        if not isinstance(val, dict):
            continue
        codes = val.get("codes", [])
        if not isinstance(codes, list):
            continue
        for code in codes:
            if isinstance(code, dict) and to_bool(code["score"]):
                return True
    return False


def build_sorted_clusters(
    clusters_payload: Dict[str, Any], inter_strategy: str, intra_strategy: str
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Returns list of (cluster_id, cluster_dict) sorted per inter-cluster strategy.
    Filters out entries without 'codes' list or with empty codes list.
    Codes within each cluster are ordered per intra-cluster strategy.
    """
    items: List[Tuple[str, Dict[str, Any]]] = []
    for cid, cval in clusters_payload.items():
        if not isinstance(cval, dict):
            continue
        codes = cval.get("codes", [])
        if not isinstance(codes, list) or not codes:
            continue
        # Order codes based on intra_strategy
        filtered_codes = [c for c in codes if isinstance(c, dict)]
        if intra_strategy == "longest":
            sorted_codes = sorted(
                filtered_codes,
                key=lambda c: to_int(c.get("tokens", 0), 0),
                reverse=True,
            )
        elif intra_strategy == "score":
            # Sort by per-solution tournament_score desc; tie-break by tokens desc
            sorted_codes = sorted(
                filtered_codes,
                key=lambda c: (
                    to_float(c.get("tournament_score", 0.0), 0.0),
                    to_int(c.get("tokens", 0), 0),
                ),
                reverse=True,
            )
        elif intra_strategy == "wins":
            # Sort by per-solution tournament_wins desc; tie-break by tokens desc
            sorted_codes = sorted(
                filtered_codes,
                key=lambda c: (
                    to_int(c.get("tournament_wins", 0), 0),
                    to_int(c.get("tokens", 0), 0),
                ),
                reverse=True,
            )
        elif intra_strategy == "shortest":
            sorted_codes = sorted(
                filtered_codes,
                key=lambda c: to_int(c.get("tokens", 0), 0),
            )
        elif intra_strategy == "random":
            sorted_codes = list(filtered_codes)
            random.shuffle(sorted_codes)
        else:  # "first"
            sorted_codes = list(filtered_codes)
        if not sorted_codes:
            continue
        cval = dict(cval)
        cval["codes"] = sorted_codes
        items.append((cid, cval))

    # Helper: compute SRank per cluster based on outputs overlap and codes size
    def compute_srank_scores(items_list: List[Tuple[str, Dict[str, Any]]]) -> Dict[str, int]:
        """Compute SRank-style overlap scores between clusters."""
        outputs_map: Dict[str, List[str]] = {}
        size_map: Dict[str, int] = {}
        for cid, cdict in items_list:
            out_list = cdict.get("output", [])
            outputs_map[cid] = [str(x) for x in out_list] if isinstance(out_list, list) else []
            codes = cdict.get("codes", [])
            size_map[cid] = len(codes) if isinstance(codes, list) else 0
        srank: Dict[str, int] = {}
        for cid, _ in items_list:
            score = 0
            list_i = outputs_map.get(cid, [])
            for cj, _ in items_list:
                if cj == cid:
                    continue
                list_j = outputs_map.get(cj, [])
                # Position-aware overlap: count indices k where list_i[k] == list_j[k]
                overlap = 0
                for ai, aj in zip(list_i, list_j):
                    if ai == aj:
                        overlap += 1
                score += size_map.get(cj, 0) * overlap
            srank[cid] = score
        return srank

    # Sort clusters based on inter_strategy
    if inter_strategy == "wins":
        items.sort(key=lambda kv: to_int(kv[1].get("tournament_wins", 0), 0), reverse=True)
    elif inter_strategy == "score":
        items.sort(key=lambda kv: to_int(kv[1].get("tournament_score", 0), 0), reverse=True)
    elif inter_strategy == "size":

        def codes_len_safe(cluster_dict: Dict[str, Any]) -> int:
            """Safely count cluster code entries."""
            codes = cluster_dict.get("codes", [])
            return len(codes) if isinstance(codes, list) else 0

        items.sort(key=lambda kv: codes_len_safe(kv[1]), reverse=True)
    elif inter_strategy == "random":
        random.shuffle(items)
    else:
        # Default fallback to wins
        items.sort(key=lambda kv: to_int(kv[1].get("tournament_wins", 0), 0), reverse=True)
    return items


def compute_submission_count_for_problem(
    clusters_payload: Dict[str, Any], inter_strategy: str, intra_strategy: str
) -> int:
    """
    Implements the described round-robin selection strategy.
    Returns -1 when no solution with score == True exists anywhere.
    """
    if not any_solution_true(clusters_payload):
        return -1

    clusters = build_sorted_clusters(clusters_payload, inter_strategy, intra_strategy)
    if not clusters:
        return -1

    # next index (k-th largest) to try per cluster
    next_idx = [0] * len(clusters)
    # count attempts
    attempts = 0

    while True:
        progressed = False
        for i, (_, cval) in enumerate(clusters):
            idx = next_idx[i]
            codes = cval["codes"]
            if idx >= len(codes):
                continue
            progressed = True
            attempts += 1
            code = codes[idx]
            if to_bool(code["score"]):
                return attempts
            next_idx[i] += 1
        if not progressed:
            # No more codes available across all clusters; safety fallback
            return -1


def cluster_has_any_true(cluster_val: Dict[str, Any]) -> bool:
    """Check whether a single cluster contains at least one true-scoring solution."""
    codes = cluster_val.get("codes", [])
    if not isinstance(codes, list):
        return False
    for code in codes:
        if isinstance(code, dict) and to_bool(code["score"]):
            return True
    return False


def compute_oracle_inside_cluster_submission_count(
    clusters_payload: Dict[str, Any], inter_strategy: str, intra_strategy: str
) -> int:
    """
    Oracle-inside-cluster strategy:
    Stop as soon as we visit (ordered per strategies) a cluster that contains any True solution.
    Returns -1 if no solution True exists anywhere.
    """
    if not any_solution_true(clusters_payload):
        return -1
    clusters = build_sorted_clusters(clusters_payload, inter_strategy, intra_strategy)
    if not clusters:
        return -1
    for idx, (_, cval) in enumerate(clusters, start=1):
        if cluster_has_any_true(cval):
            return idx
    return -1


def main() -> int:
    """CLI entrypoint to compute submission count statistics over cluster files."""
    parser = argparse.ArgumentParser(
        description="Compute submission counts per problem using tournament_wins ordering and largest-token preference."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing *_cluster.jsonl files with 'codes' and 'tournament_wins' (non-recursive).",
    )
    parser.add_argument(
        "--inter-strategy",
        type=str,
        default="wins",
        choices=["wins", "score", "size", "random"],
        help="Inter-cluster ordering strategy. One of: wins, score, size, random. Default: wins.",
    )
    parser.add_argument(
        "--intra-strategy",
        type=str,
        default="longest",
        choices=["longest", "score", "wins", "shortest", "first", "random"],
        help="Intra-cluster ordering strategy. Default: longest.",
    )
    args = parser.parse_args()

    if not args.input_dir.exists():
        print(f"Error: input dir does not exist: {args.input_dir}")
        return 1
    if not args.input_dir.is_dir():
        print(f"Error: input dir is not a directory: {args.input_dir}")
        return 1

    files: List[Path] = sorted(
        [p for p in args.input_dir.glob("*_cluster.jsonl") if p.is_file()],
        key=lambda p: extract_problem_number(p.name),
    )
    if not files:
        print(f"No *_cluster.jsonl files found under {args.input_dir}")
        return 0

    # Summary counters
    num_positive = 0
    under_2 = 0
    under_3 = 0
    under_5 = 0
    under_10 = 0

    results: List[Tuple[int, int, int]] = []

    for fp in files:
        payload = load_clusters(fp)
        if not payload:
            results.append((extract_problem_number(fp.name), -1, -1))
            continue
        sc = compute_submission_count_for_problem(payload, args.inter_strategy, args.intra_strategy)
        oc = compute_oracle_inside_cluster_submission_count(payload, args.inter_strategy, args.intra_strategy)
        results.append((extract_problem_number(fp.name), sc, oc))
        if sc > 0:
            num_positive += 1
            if sc < 2:
                under_2 += 1
            if sc < 3:
                under_3 += 1
            if sc < 5:
                under_5 += 1
            if sc < 10:
                under_10 += 1

    # Print table
    results.sort(key=lambda x: x[0])
    header_cols = ["Problem", "SubmissionCount", "Oracle inside cluster"]
    problem_col_width = max(len(header_cols[0]), max((len(str(p)) for p, _, _ in results), default=1))
    sc_col_width = max(len(header_cols[1]), max((len(str(sc)) for _, sc, _ in results), default=1))
    oc_col_width = max(len(header_cols[2]), max((len(str(oc)) for _, _, oc in results), default=1))
    print(f"{header_cols[0]:<{problem_col_width}}  {header_cols[1]:<{sc_col_width}}  {header_cols[2]:<{oc_col_width}}")
    print(f"{'-' * problem_col_width}  {'-' * sc_col_width}  {'-' * oc_col_width}")
    for p, sc, oc in results:
        print(f"{p:<{problem_col_width}}  {sc:<{sc_col_width}}  {oc:<{oc_col_width}}")

    print("=" * 60)
    print("SUBMISSION COUNT SUMMARY")
    print("=" * 60)
    print(f"Positive (>0): {num_positive}")
    print(f"Under 2:       {under_2}")
    print(f"Under 3:       {under_3}")
    print(f"Under 5:       {under_5}")
    print(f"Under 10:      {under_10}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
