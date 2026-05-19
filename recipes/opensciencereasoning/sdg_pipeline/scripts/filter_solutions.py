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

from recipes.opensciencereasoning.sdg_pipeline.scripts.utils.regex_constants import COMPILED_INTERNET_PATTERNS


def extract_python_calls(serialized_output):
    calls = []
    for msg in serialized_output:
        tool_calls = msg.get("tool_calls")
        for call in tool_calls if tool_calls else []:
            fn = call.get("function", {})
            name = fn.get("name", "")
            if "python" in name.lower():
                calls.append(call)
    return calls


def extract_python_code(call):
    try:
        payload = json.loads(call["function"].get("arguments", ""))
        return payload.get("code")
    except Exception:
        return None


def uses_internet(serialized_output):
    python_calls = extract_python_calls(serialized_output)

    for call in python_calls:
        code = extract_python_code(call)
        if not code:
            continue

        for regex in COMPILED_INTERNET_PATTERNS.values():
            if regex.search(code):
                return True

    return False


LOG = logging.getLogger(__name__)


def record_passes_filters(
    record: dict,
    only_correct: bool = False,
    gen_pass_rate_bounds: Optional[Sequence[Optional[float]]] = None,
    difficulty_pass_rate_bounds: Optional[Sequence[Optional[float]]] = None,
    majority_voting_agreement_rate_bounds: Optional[Sequence[Optional[float]]] = None,
    only_samples_with_ground_truth_answer: bool = False,
    filter_internet: bool = False,
    metadata_filters: Optional[Dict[str, List[str]]] = None,
) -> tuple[bool, Optional[str]]:
    """Return (passes, filter_reason) tuple. filter_reason is set if record fails a filter."""
    metadata_filters = metadata_filters or {}
    if only_correct and not record.get("is_correct"):
        return False, "is_correct"

    gen_pass_rate = record.get("generation_model_pass_rate")
    if (
        gen_pass_rate is not None
        and gen_pass_rate_bounds
        and (gen_pass_rate_bounds[0] >= gen_pass_rate or gen_pass_rate_bounds[1] < gen_pass_rate)
    ):
        return False, "generation_model_pass_rate"

    diff_pass_rate = record.get("difficulty_model_pass_rate")
    if (
        diff_pass_rate is not None
        and difficulty_pass_rate_bounds
        and (difficulty_pass_rate_bounds[0] >= diff_pass_rate or difficulty_pass_rate_bounds[1] < diff_pass_rate)
    ):
        return False, "difficulty_model_pass_rate"

    mv_agreement_rate = record.get("majority_voting_agreement_rate")
    if (
        mv_agreement_rate is not None
        and majority_voting_agreement_rate_bounds
        and (
            majority_voting_agreement_rate_bounds[0] >= mv_agreement_rate
            or majority_voting_agreement_rate_bounds[1] < mv_agreement_rate
        )
    ):
        return False, "majority_voting_agreement_rate"

    if only_samples_with_ground_truth_answer and not record.get("expected_answer"):
        return False, "expected_answer"

    if filter_internet and uses_internet(record.get("serialized_output", [])):
        return False, "uses_internet"

    for field, allowed in metadata_filters.items():
        candidate = record.get(field)
        if candidate not in allowed:
            return False, f"metadata:{field}"
    return True, None


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
        "--only_samples_with_ground_truth_answer",
        action="store_true",
        help="Keep only samples that have a ground truth answer",
    )
    parser.add_argument(
        "--filter_internet",
        action="store_true",
        help="Filter out samples that use internet",
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
    filter_stats = {}

    metadata_filters: Dict[str, List[str]] = args.metadata_values or {}

    with open(args.input_file, "r", encoding="utf-8") as fin, open(args.output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue

            total_records += 1
            record = json.loads(line)

            passes, filter_reason = record_passes_filters(
                record,
                only_correct=args.only_correct_solutions,
                gen_pass_rate_bounds=args.generation_model_pass_rate_range,
                difficulty_pass_rate_bounds=args.difficulty_model_pass_rate_range,
                majority_voting_agreement_rate_bounds=args.majority_voting_agreement_rate_range,
                only_samples_with_ground_truth_answer=args.only_samples_with_ground_truth_answer,
                filter_internet=args.filter_internet,
                metadata_filters=metadata_filters,
            )

            if passes:
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                written_records += 1
            else:
                filter_stats[filter_reason] = filter_stats.get(filter_reason, 0) + 1

    LOG.info(
        "Filtered %s -> %s records (%.2f%% kept) into %s",
        total_records,
        written_records,
        (written_records / total_records * 100) if total_records else 0.0,
        args.output_file,
    )

    if filter_stats:
        LOG.info("Filter statistics:")
        for filter_name, count in sorted(filter_stats.items(), key=lambda x: x[1], reverse=True):
            LOG.info(f"  {filter_name}: {count} records filtered ({count / total_records * 100:.2f}%)")


if __name__ == "__main__":
    main()
