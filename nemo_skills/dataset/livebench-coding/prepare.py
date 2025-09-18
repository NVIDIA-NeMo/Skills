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

from datasets import load_dataset


def parse_data():
    data = load_dataset("livebench/coding", split="test", trust_remote_code=True)
    # Dataset({
    #     features: ['question_id', 'category', 'turns', 'question_title', 'public_test_cases', 'private_test_cases', 'original_json', 'release_date', 'citation', 'task', 'livebench_release_date', 'livebench_removal_date', 'remainder', 'solution', 'partial_solution'],
    #     num_rows: 128
    # })
    return data


def clean_data(dataset):
    def map_fn(data):
        question = data["turns"][0]
        data["question"] = question.replace("    ", "\t")
        return data

    remove_columns = [
        "category",
        "turns",
        "question_title",
        "public_test_cases",
        "private_test_cases",
        "original_json",
        "release_date",
        "citation",
        "livebench_release_date",
        "livebench_removal_date",
        "remainder",
        "solution",
    ]
    dataset = dataset.map(map_fn, remove_columns=remove_columns)
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).parent))
    args = parser.parse_args()

    data = parse_data()
    data = clean_data(data)
    print("Len of data: ", len(data))

    print("Writing to file...")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    output_file_path = os.path.join(args.output_dir, "test.jsonl")
    with open(output_file_path, "w") as f:
        for problem in data:
            json.dump(problem, f)
            f.write("\n")
