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

# ---- Fast JSON (prefer orjson) ----
import json as _json_std
import logging
import sys
from pathlib import Path
from typing import List

from transformers import AutoTokenizer

try:  # pragma: no cover - best effort
    import orjson as _orjson  # type: ignore

    def _json_loads(s: str):
        return _orjson.loads(s)

    def _json_dumps(obj) -> str:
        return _orjson.dumps(obj).decode("utf-8")

except Exception:  # pragma: no cover
    _orjson = None

    def _json_loads(s: str):
        return _json_std.loads(s)

    def _json_dumps(obj) -> str:  # type: ignore
        return _json_std.dumps(obj, ensure_ascii=False)


# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------


def compute_token_length(text: str, tokenizer: AutoTokenizer) -> int:
    """Compute number of tokens for a given text."""
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def bucket_index(length: int, bucket_sizes: List[int]) -> int:
    """Return bucket index (upper bound) for given length."""
    for size in bucket_sizes:
        if length <= size:
            return size
    return -1  # overflow


# ------------------------------------------------------------
# Core logic
# ------------------------------------------------------------


def process_jsonl(
    input_path: str,
    output_dir: str,
    tokenizer: AutoTokenizer,
    to_bucket: bool = False,
    bucket_sizes: List[int] = None,
):
    """Add out_token_length, optionally bucket by token size."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if to_bucket and not bucket_sizes:
        raise ValueError("If to_bucket=True, you must provide bucket_sizes (list of ints).")

    # Prepare bucket files
    bucket_files = {}
    bucket_counts = {}
    if to_bucket:
        for b in bucket_sizes:
            bucket_path = output_dir / f"{input_path.stem}_bucket_{b}.jsonl"
            bucket_files[b] = open(bucket_path, "w", encoding="utf-8")
            bucket_counts[b] = 0
        overflow_path = output_dir / f"{input_path.stem}_bucket_overflow.jsonl"
        bucket_files["overflow"] = open(overflow_path, "w", encoding="utf-8")
        bucket_counts["overflow"] = 0
    else:
        output_path = output_dir / f"{input_path.stem}_with_tokens.jsonl"
        outfile = open(output_path, "w", encoding="utf-8")

    processed = 0
    with open(input_path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json_loads(line)
                out_text = obj.get("output", "")
                if "out_token_length" not in obj:
                    length = compute_token_length(out_text, tokenizer)
                    obj["out_token_length"] = length
                    dumped = _json_dumps(obj)

                else:
                    length = obj["out_token_length"]
                    dumped = line  # already has the field

                if to_bucket:
                    b = bucket_index(length, bucket_sizes)
                    if b != -1:
                        bucket_files[b].write(dumped + "\n")
                        bucket_counts[b] += 1
                    else:
                        bucket_files["overflow"].write(dumped + "\n")
                        bucket_counts["overflow"] += 1
                else:
                    outfile.write(dumped + "\n")

                processed += 1
                if processed % 1000 == 0:
                    logging.info(f"Processed {processed} lines")

            except Exception as e:
                logging.error(f"Error processing line: {e}")

    if to_bucket:
        for f in bucket_files.values():
            f.close()
    else:
        outfile.close()

    # ---- Summary logging ----
    logging.info(f"✅ Done! Processed {processed} examples total.")
    if to_bucket:
        logging.info("📊 Bucket distribution:")
        total = sum(bucket_counts.values())
        for b, count in bucket_counts.items():
            pct = (count / total * 100) if total > 0 else 0
            logging.info(f"  - Bucket {b:<10}: {count:>8} items ({pct:.1f}%)")
        logging.info(f"💾 Saved bucketed files in: {output_dir}")
    else:
        logging.info(f"💾 Saved to {output_path}")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Add output token length and optionally bucket by token size.")
    parser.add_argument("input_file", type=str, help="Path to input .jsonl file")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save processed files")
    parser.add_argument("--to_bucket", action="store_true", help="Whether to bucket results by token length")
    parser.add_argument(
        "--bucket_sizes",
        default=[16000, 32000, 64000],
        nargs="+",
        type=int,
        help="List of bucket size upper limits (e.g. 16000, 32000, 64000)",
    )
    parser.add_argument("--tokenizer_path", type=str, help="Model name for tokenizer")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    process_jsonl(args.input_file, args.output_dir, tokenizer, args.to_bucket, args.bucket_sizes)
