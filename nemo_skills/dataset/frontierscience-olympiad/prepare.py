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

import argparse
import json
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

HF_REPO = "openai/frontierscience"
HF_CONFIG = "olympiad"
HF_SPLIT = "test"
HF_DATA_FILE = "olympiad/test.jsonl"

# Map of available subjects
SUBJECTS = ["chemistry", "biology", "physics"]


def format_entry(entry, problem_index):
    """Format entry for nemo-skills from FrontierScience Olympiad dataset."""
    answer = entry.get("answer", "")
    # Remove surrounding backticks (handles `, ``, ```, etc.)
    answer = re.sub(r"^`+|`+$", "", answer).strip()

    formatted = {
        "id": f"olympiad-{problem_index}",
        "question": entry.get("problem", ""),
        "expected_answer": answer,
        "subset_for_metrics": entry.get("subject", ""),
        "task_group_id": entry.get("task_group_id", ""),
    }

    return formatted


def write_data_to_file(output_file, data, subject_filter=None):
    """Write formatted data to JSONL file."""
    count = 0
    with open(output_file, "wt", encoding="utf-8") as fout:
        for idx, entry in enumerate(tqdm(data, desc=f"Writing {output_file.name}")):
            # Filter by subject if specified
            if subject_filter and entry.get("subject", "").lower() != subject_filter:
                continue
            formatted_entry = format_entry(entry, idx)
            json.dump(formatted_entry, fout)
            fout.write("\n")
            count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="all",
        choices=["all"] + SUBJECTS,
        help="Dataset split to process (all/chemistry/biology/physics).",
    )
    args = parser.parse_args()

    # Load the FrontierScience olympiad dataset from HuggingFace datasets API
    print(
        "Loading FrontierScience olympiad dataset from HuggingFace datasets "
        f"({HF_REPO}, {HF_CONFIG}, split={HF_SPLIT})..."
    )
    olympiad_data = load_dataset(
        HF_REPO,
        data_files={HF_SPLIT: HF_DATA_FILE},
        split=HF_SPLIT,
    )

    print(f"Loaded {len(olympiad_data)} olympiad problems")

    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)

    if args.split == "all":
        applied_subjects = SUBJECTS
    else:
        applied_subjects = [args.split]
    # Process all subjects separately
    for subject in applied_subjects:
        output_file = data_dir / f"{subject}.jsonl"
        count = write_data_to_file(output_file, olympiad_data, subject_filter=subject)
        print(f"Saved {count} {subject} entries to {output_file}")
    if args.split == "all":
        # Also create a combined all.jsonl with all problems
        output_file = data_dir / "all.jsonl"
        count = write_data_to_file(output_file, olympiad_data)
        print(f"Saved {count} total entries to {output_file}")
