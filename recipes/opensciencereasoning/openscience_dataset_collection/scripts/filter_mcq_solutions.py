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
"""
Filter generations and keep only the majority-voted answer for each question.

Outputs two files in <output_dir>:
1. filtered.jsonl      — one consolidated entry per question with the consensus generation.
2. filtered_all.jsonl  — all entries matching the consensus answer for each question.
"""

import argparse
import glob
import json
import logging
import os
import re
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from pprint import pprint
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm


def extract_answer(
    string: str,
    extract_from_boxed: bool = True,
    extract_regex: str = r"The final answer is (.+)$",
):
    """Extract Answer String from \\boxed expression or based on regex"""
    if not extract_from_boxed:
        match = re.search(extract_regex, string)
        if match:
            return match.group(1)
        return None

    if "\\boxed" not in string:
        return None

    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    if retval:
        left = "\\boxed{"
        try:
            assert retval[: len(left)] == left
            assert retval[-1] == "}"
            return retval[len(left) : -1]
        except AssertionError:
            return None

    return None


def get_answer_after_think(text):
    return extract_answer(text.split("</think>")[-1])


def has_thought(gen: str) -> bool:
    """Return True if gen contains exactly one <think>...</think>."""
    return gen.count("<think>") == 1 and gen.count("</think>") == 1


def process_prediction_group(
    records: List[Dict[str, Any]], min_samples: int, majority_threshold: float, generation_field: str
) -> Optional[Tuple[List[Dict[str, Any]], str, int, int]]:
    # Keep only entries with a valid thought block and boxed answer
    valid_records = [
        r for r in records if has_thought(r[generation_field]) and get_answer_after_think(r[generation_field])
    ]
    if not valid_records:
        return None

    # Count occurrences of each answer letter
    counts: Dict[str, int] = defaultdict(int)
    for rec in valid_records:
        letter = get_answer_after_think(rec[generation_field])
        if letter:
            counts[letter] += 1
    if not counts:
        return None

    # Determine the most common answer
    consensus_letter = max(sorted(counts.items()), key=lambda kv: kv[1])[0]
    # Select all records matching this consensus
    consensus_records = [r for r in valid_records if get_answer_after_think(r[generation_field]) == consensus_letter]
    if len(consensus_records) < min_samples or sum(counts.values()) * majority_threshold > len(consensus_records):
        return None

    return (
        consensus_records,
        consensus_letter,
        len(consensus_records),
        len(valid_records),
    )


def main() -> None:
    # Set up logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Filter by majority vote per question.")
    parser.add_argument(
        "-i",
        "--input_path",
        required=True,
        help="Directory or glob pattern for .jsonl prediction files",
    )
    parser.add_argument("-o", "--output_dir", required=True, help="Directory to save filtered outputs")
    parser.add_argument(
        "--meta_keys",
        required=False,
        default=[],
        nargs="+",
    )
    parser.add_argument(
        "--majority_threshold",
        type=float,
        default=0,
        help="Minimum fraction of consensus votes required to accept a majority decision",
    )
    parser.add_argument(
        "--min_samples", type=int, default=0, help="Minimum number of samples required to perform majority voting"
    )
    parser.add_argument("--file_name", type=str, default="filtered", help="Name of the output file")
    parser.add_argument(
        "--question_field",
        type=str,
        default="question",
        help="Name of the field in the input data containing the question",
    )
    parser.add_argument(
        "--generation_field",
        type=str,
        default="generation",
        help="Name of the field in the input data containing the model's generated response",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find all JSONL files
    pattern = os.path.join(args.input_path.rstrip(os.sep), "*.jsonl")
    files = sorted(glob.glob(pattern))
    if not files:
        logging.error("No JSONL files found in %s", args.input_path)
        return
    logging.info("Found %d files to process.", len(files))

    # Group by question text
    question_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for fp in tqdm(files, desc="Reading files", unit="file"):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                question_groups[data[args.question_field]].append(data)
    logging.info("Collected %d unique questions.", len(question_groups))

    # Process groups in parallel
    results: List[Tuple[List[Dict[str, Any]], str, int, int]] = []
    with Pool(cpu_count()) as pool:
        tasks = [
            pool.apply_async(
                process_prediction_group, (grp, args.min_samples, args.majority_threshold, args.generation_field)
            )
            for grp in question_groups.values()
        ]
        question_groups.clear()  # free memory
        for task in tqdm(tasks, desc="Processing groups", total=len(tasks), unit="group"):
            out = task.get()
            if out:
                results.append(out)

    # Write filtered outputs
    only_one_path = os.path.join(args.output_dir, f"{args.file_name}.jsonl")
    all_path = os.path.join(args.output_dir, f"{args.file_name}_all.jsonl")
    was_warned_missing = set()
    with open(only_one_path, "w", encoding="utf-8") as f1, open(all_path, "w", encoding="utf-8") as f2:
        for recs, letter, cnt, total in tqdm(results, desc="Writing files", unit="entry"):
            # Write a single entry per question
            first = recs[0]
            entry = {
                "problem": first[args.question_field],
                "generation": first[args.generation_field],
                "expected_answer": letter,
                "majority_res": f"{cnt}/{total}",
            }
            for key in args.meta_keys:
                if key in first:
                    entry[key] = first[key]
                elif key not in was_warned_missing:
                    logging.warning("Key %s not found in record: %s", key, first)
                    was_warned_missing.add(key)
            f1.write(json.dumps(entry, ensure_ascii=False) + "\n")
            # Write all consensus-matching entries
            for r in recs:
                entry_all = {
                    "problem": r[args.question_field],
                    "generation": r[args.generation_field],
                    "expected_answer": letter,
                    "majority_res": f"{cnt}/{total}",
                }
                for key in args.meta_keys:
                    if key in r:
                        entry_all[key] = r[key]
                    elif key not in was_warned_missing:
                        logging.warning("Key %s not found in record: %s", key, r)
                        was_warned_missing.add(key)
                f2.write(json.dumps(entry_all, ensure_ascii=False) + "\n")
    # Print one example from the single-entry file
    try:
        with open(only_one_path, "r", encoding="utf-8") as f:
            example = json.loads(f.readline())
        logging.info("Sample entry:")
        pprint(example)
    except Exception as e:
        logging.warning("Failed to print sample entry: %s", e)

    logging.info("Wrote one entry per question to %s", only_one_path)
    logging.info("Wrote all matching entries to %s", all_path)


if __name__ == "__main__":
    main()
