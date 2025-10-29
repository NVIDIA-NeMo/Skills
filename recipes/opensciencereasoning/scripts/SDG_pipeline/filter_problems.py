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

# --- Fast JSON ---
import json as _json_std
import logging
import re
import sys
from pathlib import Path

try:
    import orjson as _orjson  # type: ignore

    def _json_loads(s: str):
        return _orjson.loads(s)

    def _json_dumps(obj) -> str:
        return _orjson.dumps(obj).decode("utf-8")
except Exception:
    _orjson = None

    def _json_loads(s: str):
        return _json_std.loads(s)

    def _json_dumps(obj) -> str:
        return _json_std.dumps(obj, ensure_ascii=False)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------


def extract_dataset_name(input_path: str) -> str:
    return Path(input_path).stem


def generate_id(dataset_name: str, line_index: str) -> str:
    """Generate a unique id."""
    return f"{dataset_name}--{str(line_index).zfill(6)}"


def count_options(problem):
    """
    Count how many options are present based on typical MCQ formatting.
    Assumes options start with 'A) ', 'B) ', etc. and are separated by double newlines.
    Supported format: 'Question\n\nA) option text'\nB) option text\n...
    """
    parts = problem.split("\n\n")
    options_part = None
    for part in parts:
        if part.startswith("A) "):
            options_part = part
            break
    if not options_part:
        return 0

    for i, option in enumerate(options_part.split("\n")):
        if not option.startswith(f"{chr(ord('A') + i)}) "):
            return i
    return i + 1


def match_option_format(text: str, pattern: str) -> bool:
    """Check if option format matches given regex pattern at least once."""
    return bool(re.search(pattern, text))


def contains_image(problem: str) -> bool:
    extensions = [".png", ".jpg", ".jpeg", ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx", ".svg"]
    if any(ext in problem for ext in extensions):
        return True
    return False


# ---------------------------------------------------
# Main filtering function
# ---------------------------------------------------


def process_file(
    input_file: str,
    output_file: str,
    remove_images: bool = False,
    deduplicate: bool = False,
    dataset_name: str = None,
    num_options: int | None = None,
    option_format_regex: str = None,
    problem_field: str = "problem",
    expected_answer_field: str = "expected_answer",
    id_field: str = "id",
    remove_expected_answer: bool = False,
):
    input_file = Path(input_file)
    output_file = Path(output_file)

    if dataset_name is None:
        dataset_name = extract_dataset_name(str(input_file))

    seen_problems = set()
    kept = 0
    dropped = 0

    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for index, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json_loads(line)
            except Exception as e:
                logging.warning(f"⚠️ Could not parse line: {e}")
                dropped += 1
                continue

            # rename keys to standard names
            for current_key, new_key in [
                (problem_field, "problem"),
                (expected_answer_field, "expected_answer"),
                (id_field, "id"),
            ]:
                if current_key in obj and new_key != current_key:
                    obj[new_key] = obj[current_key]
                    del obj[current_key]

            if remove_expected_answer and "expected_answer" in obj:
                del obj["expected_answer"]

            problem = obj.get("problem", "")
            if not problem or not problem.strip():
                dropped += 1
                continue

            # ensure id
            if "id" not in obj:
                logging.warning("⚠️ Missing id, generating one.")
                obj["id"] = generate_id(dataset_name, index)

            # deduplicate
            if deduplicate:
                if problem in seen_problems:
                    dropped += 1
                    continue
                seen_problems.add(problem)

            # filter images
            if remove_images and contains_image(problem):
                dropped += 1
                continue

            # filter by number of options
            if num_options is not None:
                option_count = count_options(problem)
                if option_count != num_options:
                    dropped += 1
                    continue

            # filter by option format regex
            if option_format_regex is not None:
                if not match_option_format(problem, option_format_regex):
                    dropped += 1
                    continue

            # add everything beside id problem and expected_answer to metadata
            metadata = {k: v for k, v in obj.items() if k not in ["id", "problem", "expected_answer"]}
            obj["metadata"] = metadata
            # remove old metadata keys
            for key in metadata.keys():
                del obj[key]

            # write output
            fout.write(_json_dumps(obj) + "\n")
            kept += 1

    total = kept + dropped
    preservation_pct = (kept / total * 100) if total > 0 else 0

    logging.info(f"✅ Total datapoints processed: {total}")
    logging.info(f"✅ Kept: {kept}")
    logging.info(f"🗑️ Dropped: {dropped}")
    logging.info(f"📊 Preservation: {preservation_pct:.2f}%")
    logging.info(f"💾 Saved filtered file to: {output_file}")


# ---------------------------------------------------
# Entry point
# ---------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter JSONL dataset by question content and MCQ option format.")
    parser.add_argument("input_file", type=str, help="Path to input JSONL file")
    parser.add_argument("output_file", type=str, help="Path to output JSONL file")
    parser.add_argument("--deduplicate", action="store_true", help="Remove duplicate problems")
    parser.add_argument(
        "--dataset_name", type=str, help="Dataset name (optional). If not provided, derived from filename"
    )
    parser.add_argument("--num_options", type=int, default=None, help="Filter by number of options")
    parser.add_argument("--remove_images", action="store_true", help="Remove images from problems")
    parser.add_argument("--option_format_regex", type=str, help="Filter by option format regex (e.g. '^[A-Z]\\)')")
    parser.add_argument("--problem_field", type=str, help="Field in the JSONL file that contains the problem")
    parser.add_argument(
        "--expected_answer_field", type=str, help="Field in the JSONL file that contains the expected answer"
    )
    parser.add_argument("--remove_expected_answer", action="store_true", help="Remove expected answer from samples")
    parser.add_argument("--id_field", type=str, help="Field in the JSONL file that contains the id")

    args = parser.parse_args()

    process_file(
        args.input_file,
        args.output_file,
        deduplicate=args.deduplicate,
        dataset_name=args.dataset_name,
        num_options=args.num_options,
        option_format_regex=args.option_format_regex,
        problem_field=args.problem_field,
        expected_answer_field=args.expected_answer_field,
        id_field=args.id_field,
        remove_expected_answer=args.remove_expected_answer,
    )
