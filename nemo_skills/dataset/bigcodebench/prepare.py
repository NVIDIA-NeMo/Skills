# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import os
from pathlib import Path

import datasets

BIGCODEBENCH_VERSION = "v0.1.4"


def parse_data(split="hard"):
    dataset_name = "bigcode/bigcodebench" if split == "full" else "bigcode/bigcodebench-hard"
    data = datasets.load_dataset(dataset_name, split=BIGCODEBENCH_VERSION)
    return data


def extract_prefix(text: str, delimiter: str) -> str:
    index = text.find(delimiter)
    assert index != -1
    return text[:index].strip()


def clean_data(dataset, subset):
    def map_fn(data):
        prefix = extract_prefix(data["instruct_prompt"], "You should write self-contained code starting with:")
        code_prompt = wrap_in_code_tag(data["code_prompt"])
        data["question"] = prefix + "\n\n" + "You should write self-contained code starting with:" + "\n" + code_prompt
        return data

    if subset == "hard":
        remove_columns = [
            "_id",
            "complete_prompt",
            "instruct_prompt",
            "canonical_solution",
            "code_prompt",
            "test",
            "entry_point",
            "doc_struct",
            "libs",
            "q_idx",
            "score",
        ]
    else:
        remove_columns = [
            "complete_prompt",
            "instruct_prompt",
            "canonical_solution",
            "code_prompt",
            "test",
            "entry_point",
            "doc_struct",
            "libs",
        ]
    dataset = dataset.map(map_fn, remove_columns=remove_columns)
    return dataset


def wrap_in_code_tag(text):
    if "```" not in text or "```python" not in text:
        return f"```python\n{text}\n```"
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).parent))
    parser.add_argument("--split", type=str, default="full", choices=["full", "hard"])

    args = parser.parse_args()

    data = parse_data(split=args.split)
    data = clean_data(data, args.split)
    print("Len of data: ", len(data))

    print("Writing to file...")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_file_path = os.path.join(args.output_dir, f"{args.split}.jsonl")
    with open(output_file_path, "w") as f:
        for problem in data:
            # somehow models like tabs more than spaces
            problem["question"] = problem["question"].replace("    ", "\t")
            problem["split"] = args.split
            json.dump(problem, f)
            f.write("\n")
