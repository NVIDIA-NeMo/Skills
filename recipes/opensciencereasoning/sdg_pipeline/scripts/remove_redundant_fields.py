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

"""Prepare inputs for the difficulty estimation stage."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

LOG = logging.getLogger(__name__)


def process_file(
    input_path: Path,
    output_path: Path,
    fields: List[str],
    deduplicate: bool = False,
    deduplicate_by: str = "problem",
) -> None:
    """
    Remove redundant fields from the input file and optionally deduplicate.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        fields: List of fields to keep
        deduplicate: Whether to deduplicate entries
        deduplicate_by: Field to use for deduplication (default: "problem")
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen = set()
    count_total = 0
    count_written = 0

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            count_total += 1
            data = json.loads(line)

            # Deduplicate if requested
            if deduplicate:
                if deduplicate_by not in data:
                    LOG.warning(
                        f"Deduplication field '{deduplicate_by}' not found in entry, skipping deduplication for this line"
                    )
                else:
                    dedup_key = data[deduplicate_by]
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

            # Keep only specified fields
            sample = {key: value for key, value in data.items() if key in fields}
            fout.write(json.dumps(sample) + "\n")
            count_written += 1

    LOG.info(f"Processed {count_total} entries, wrote {count_written} entries")
    if deduplicate:
        LOG.info(f"Removed {count_total - count_written} duplicate entries")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare difficulty estimation inputs by keeping core fields and optionally deduplicating."
    )
    parser.add_argument("--input_file", required=True, type=Path, help="Path to the JSONL input file")
    parser.add_argument("--output_file", required=True, type=Path, help="Path to write the prepared JSONL")
    parser.add_argument("--fields", required=True, type=json.loads, help="List of fields to keep (JSON list)")
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Deduplicate entries based on a field (default: disabled)",
    )
    parser.add_argument(
        "--deduplicate_by",
        type=str,
        default="problem",
        help="Field to use for deduplication (default: 'problem')",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if not args.input_file.exists():
        raise FileNotFoundError(f"Input file does not exist: {args.input_file}")

    LOG.info("Reading input from %s", args.input_file)
    if args.deduplicate:
        LOG.info(f"Deduplicating entries by field: '{args.deduplicate_by}'")

    process_file(
        args.input_file,
        args.output_file,
        args.fields,
        deduplicate=args.deduplicate,
        deduplicate_by=args.deduplicate_by,
    )


if __name__ == "__main__":
    main()
