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
Script to filter clusters by removing candidates with sample_score=False.
Removes empty clusters and updates the status field.
Output is written to clusters_filtered directory.
"""

import argparse
import json
import sys
from pathlib import Path


def filter_cluster(cluster_data):
    """
    Filter codes in a cluster, keeping only those with sample_score=True.

    Args:
        cluster_data: Dictionary containing cluster information

    Returns:
        Filtered cluster data, or None if no codes remain
    """
    if "codes" not in cluster_data:
        return None

    # Filter codes to keep only those with sample_score=True
    filtered_codes = [code for code in cluster_data["codes"] if code.get("sample_score", False)]

    # If no codes remain after filtering, return None (cluster will be removed)
    if not filtered_codes:
        return None

    # Create filtered cluster with updated codes and status
    filtered_cluster = cluster_data.copy()
    filtered_cluster["codes"] = filtered_codes

    # Update status field with pass/fail counts
    test_passed = sum(1 for code in filtered_codes if bool(code.get("score", False)))
    test_failed = len(filtered_codes) - test_passed
    sample_passed = sum(1 for code in filtered_codes if bool(code.get("sample_score", False)))
    sample_failed = len(filtered_codes) - sample_passed

    filtered_cluster["status"] = {
        "Test passed": test_passed,
        "Test failed": test_failed,
        "Sample passed": sample_passed,
        "Sample failed": sample_failed,
    }

    return filtered_cluster


def filter_file(input_file, output_file):
    """
    Filter all clusters in a file.

    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file

    Returns:
        Tuple of (original_cluster_count, filtered_cluster_count, original_code_count, filtered_code_count)
    """
    # Read input file
    with open(input_file, "r") as f:
        data = json.load(f)

    # Filter each cluster
    filtered_data = {}
    original_cluster_count = 0
    original_code_count = 0
    filtered_code_count = 0

    for cluster_key, cluster_value in data.items():
        if cluster_key.startswith("cluster_"):
            original_cluster_count += 1
            if "codes" in cluster_value:
                original_code_count += len(cluster_value["codes"])

            filtered_cluster = filter_cluster(cluster_value)

            if filtered_cluster is not None:
                filtered_data[cluster_key] = filtered_cluster
                filtered_code_count += len(filtered_cluster["codes"])

    # Write output file
    with open(output_file, "w") as f:
        json.dump(filtered_data, f, indent=4)

    filtered_cluster_count = len(filtered_data)

    return (
        original_cluster_count,
        filtered_cluster_count,
        original_code_count,
        filtered_code_count,
    )


def main():
    """CLI entrypoint for filtering cluster files by sample_score."""
    parser = argparse.ArgumentParser(description="Filter clusters by removing candidates with sample_score=False.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Input directory containing *_cluster.jsonl files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory to write filtered cluster files",
    )
    args = parser.parse_args()
    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Directory '{input_dir}' not found!")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created output directory: {output_dir}\n")

    # Find all cluster files
    cluster_files = sorted(input_dir.glob("*_cluster.jsonl"), key=lambda x: int(x.stem.split("_")[0]))

    if not cluster_files:
        print(f"Error: No *_cluster.jsonl files found in '{input_dir}'!")
        sys.exit(1)

    # Process each file
    print("Filtering clusters...\n")
    total_original_clusters = 0
    total_filtered_clusters = 0
    total_original_codes = 0
    total_filtered_codes = 0

    for input_file in cluster_files:
        output_file = output_dir / input_file.name

        orig_clusters, filt_clusters, orig_codes, filt_codes = filter_file(input_file, output_file)

        total_original_clusters += orig_clusters
        total_filtered_clusters += filt_clusters
        total_original_codes += orig_codes
        total_filtered_codes += filt_codes

        removed_clusters = orig_clusters - filt_clusters
        removed_codes = orig_codes - filt_codes

        print(f"{input_file.name}:")
        print(f"  Clusters: {orig_clusters} -> {filt_clusters} (removed {removed_clusters} empty)")
        print(f"  Codes:    {orig_codes} -> {filt_codes} (removed {removed_codes} with sample_score=False)")
        print()

    # Print summary
    print("=" * 70)
    print("SUMMARY:")
    print(
        f"  Total clusters: {total_original_clusters} -> {total_filtered_clusters} "
        f"(removed {total_original_clusters - total_filtered_clusters} empty)"
    )
    print(
        f"  Total codes:    {total_original_codes} -> {total_filtered_codes} "
        f"(removed {total_original_codes - total_filtered_codes} with sample_score=False)"
    )
    print(f"\nFiltered files saved to: {output_dir}/")


if __name__ == "__main__":
    main()
