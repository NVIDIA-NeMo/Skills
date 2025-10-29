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

"""Filter aggregated solution records according to quality thresholds."""

import argparse
import json
import logging
import os
from typing import Dict, List, Optional, Sequence

LOG = logging.getLogger(__name__)


def record_passes_filters(
    record: dict,
    only_correct: bool = False,
    gen_pass_rate_bounds: Optional[Sequence[Optional[float]]] = None,
    difficulty_pass_rate_bounds: Optional[Sequence[Optional[float]]] = None,
    majority_voting_agreement_rate_bounds: Optional[Sequence[Optional[float]]] = None,
    is_ground_truth_answer_present: bool = False,
    metadata_filters: Optional[Dict[str, List[str]]] = None,
) -> bool:
    """Return True when a record satisfies correctness, pass-rate, and metadata constraints."""
    metadata_filters = metadata_filters or {}
    if only_correct and not record.get("is_correct"):
        return False
    if gen_pass_rate_bounds and (
        gen_pass_rate_bounds[0] >= record["generation_model_pass_rate"]
        or gen_pass_rate_bounds[1] < record["generation_model_pass_rate"]
    ):
        return False
    if difficulty_pass_rate_bounds and (
        difficulty_pass_rate_bounds[0] >= record["difficulty_model_pass_rate"]
        or difficulty_pass_rate_bounds[1] < record["difficulty_model_pass_rate"]
    ):
        return False
    if majority_voting_agreement_rate_bounds and (
        majority_voting_agreement_rate_bounds[0] >= record["majority_voting_agreement_rate"]
        or majority_voting_agreement_rate_bounds[1] < record["majority_voting_agreement_rate"]
    ):
        return False
    if is_ground_truth_answer_present and not record["expected_answer"]:
        return False

    for field, allowed in metadata_filters.items():
        candidate = record.get(field)
        if candidate not in allowed:
            return False

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse CLI arguments for filtering by correctness, pass rates, and metadata."
    )
    parser.add_argument("--input_file", required=True, help="Path to final_result.jsonl before filtering")
    parser.add_argument("--output_file", required=True, help="Where to write filtered records")
    parser.add_argument(
        "--only_correct_solutions",
        action="store_true",
        help="Keep only samples whose 'is_correct' flag is True",
    )
    parser.add_argument(
        "--generation_model_pass_rate_range",
        type=json.loads,
        default=None,
        help="JSON array [min, max] (min exclusive, max inclusive) for generation_model_pass_rate",
    )
    parser.add_argument(
        "--difficulty_model_pass_rate_range",
        type=json.loads,
        default=None,
        help="JSON array [min, max] (min exclusive, max inclusive) for pass_rate",
    )
    parser.add_argument(
        "--majority_voting_agreement_rate_range",
        type=json.loads,
        default=None,
        help="JSON array [min, max] (min exclusive, max inclusive) for majority_voting_agreement_rate",
    )
    parser.add_argument(
        "--is_ground_truth_answer_present",
        action="store_true",
        help="Keep only samples whose ground truth answer is present",
    )
    parser.add_argument(
        "--metadata_values",
        type=json.loads,
        default={},
        help="JSON dict mapping metadata fields to allowed string values",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point that applies filters to the aggregated solutions file."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if not os.path.exists(args.input_file):
        raise FileNotFoundError(f"Input file not found: {args.input_file}")

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)

    total_records = 0
    written_records = 0

    metadata_filters: Dict[str, List[str]] = args.metadata_values or {}

    with open(args.input_file, "r", encoding="utf-8") as fin, open(args.output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            total_records += 1
            record = json.loads(line)

            if record_passes_filters(
                record,
                only_correct=args.only_correct_solutions,
                gen_pass_rate_bounds=args.generation_model_pass_rate_range,
                difficulty_pass_rate_bounds=args.difficulty_model_pass_rate_range,
                majority_voting_agreement_rate_bounds=args.majority_voting_agreement_rate_range,
                is_ground_truth_answer_present=args.is_ground_truth_answer_present,
                metadata_filters=metadata_filters,
            ):
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written_records += 1

    LOG.info(
        "Filtered %s -> %s records (%.2f%% kept) into %s",
        total_records,
        written_records,
        (written_records / total_records * 100) if total_records else 0.0,
        args.output_file,
    )


if __name__ == "__main__":
    main()
