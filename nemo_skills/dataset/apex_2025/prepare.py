# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

HF_REPO_ID = "MathArena/apex_2025"


def format_entry(entry):
    return {
        "problem_idx": entry["problem_idx"],
        "source": entry["source"],
        "problem": entry["problem"],
        "expected_answer": str(entry["answer"]),
    }


def write_data_to_file(output_file, data):
    formatted_entries = [format_entry(entry) for entry in tqdm(data, desc=f"Formatting {output_file.name}")]

    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in tqdm(formatted_entries, desc=f"Writing {output_file.name}"):
            json.dump(entry, fout, ensure_ascii=False)
            fout.write("\n")


def main(args):
    dataset = load_dataset(HF_REPO_ID, split="train")
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / f"{args.split}.jsonl"
    write_data_to_file(output_file, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("test",), help="Dataset split to process.")
    args = parser.parse_args()
    main(args)
