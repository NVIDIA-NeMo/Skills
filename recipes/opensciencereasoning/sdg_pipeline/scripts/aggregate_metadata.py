#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Dict, List


def read_jsonl(path: str) -> List[dict]:
    records: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def merge_metadata(metadata_files: List[str]) -> Dict[str, dict]:
    """Merge all keys per problem across provided JSONL files.

    If the same key appears for the same problem in multiple files, the last
    occurrence wins (based on the order in metadata_files).
    """
    by_problem = defaultdict(dict)
    for path in metadata_files:
        for sample in read_jsonl(path):
            by_problem[sample["problem"]].update(sample)
    return by_problem


def collect_solutions(solutions_path: str) -> List[dict]:
    """Collect solutions and optional judgements per problem.
    Multiple files (e.g., output-rs*.jsonl) are supported and concatenated.
    """

    solutions = []
    for path in glob.glob(solutions_path):
        solutions.extend(read_jsonl(path))

    return solutions


def write(output_file: str, dataset: List[dict], metadata: Dict[str, dict]):
    """Write dataset with metadata."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fout:
        for sample in dataset:
            sample.update(metadata[sample["problem"]])
            fout.write(json.dumps(sample) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate results: if solutions exist, use them as base and overlay metadata; "
            "otherwise, use metadata (first file defines base problems)."
        )
    )
    parser.add_argument("--output_file", required=True, type=str)
    parser.add_argument(
        "--metadata_files", required=False, default="[]", type=json.loads, help="JSON list of .jsonl files"
    )
    parser.add_argument("--solutions_path", required=False, default=None, type=str)
    args = parser.parse_args()

    if not args.solutions_path and (not isinstance(args.metadata_files, list) or len(args.metadata_files) == 0):
        raise ValueError("Provide at least --solutions_path or --metadata_files (one of them is required)")

    metadata: Dict[str, dict] = {}
    if args.metadata_files:
        metadata = merge_metadata(args.metadata_files)

    solutions: List[dict] = []
    if args.solutions_path:
        solutions = collect_solutions(args.solutions_path)

    if solutions:
        dataset = solutions
    else:
        dataset = list(metadata.values())

    write(args.output_file, dataset, metadata)


if __name__ == "__main__":
    main()
