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

import json
import os
from pathlib import Path

from datasets import Value, load_dataset


class PromptConstants:
    # reference: https://huggingface.co/nvidia/Nemotron-Cascade-8B/blob/main/evaluation/data/benchmark.py#L1166
    FORMATTING_WITHOUT_STARTER_CODE = (
        "Write a C++14 or C++17 code to solve the problem. "
        "Please place the solution code in the following format:\n"
        "```cpp\n// Your solution code here\n```"
    )
    FORMATTING_MESSAGE_WITH_STARTER_CODE = (
        "Please place the solution code in C++14 or C++17 language in the following format:\n"
        "```cpp\n// Your solution code here\n```"
    )


def parse_data(split):
    data = load_dataset("nvidia/LiveCodeBench-CPP", split=split)

    # data has the following fields
    # question_title: str
    # question_content: str
    # platform: Platform
    # question_id: str
    # contest_id: str
    # contest_date: datetime
    # starter_code: str
    # difficulty: Difficulty
    # public_test_cases: list[Test]
    # private_test_cases: list[Test]
    # metadata: dict
    return data


def clean_data(dataset, keep_all_columns=False):
    def map_fn(data):
        if data["starter_code"]:
            data["starter_code"] = (
                "\n\n"
                + "Solve the problem starting with the provided function header.\n\nFunction header:\n"
                + "```\n"
                + data["starter_code"]
                + "\n```"
            )
            data["formatting_message"] = PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE
        else:
            data["starter_code"] = ""
            data["formatting_message"] = PromptConstants.FORMATTING_WITHOUT_STARTER_CODE

        data["task_id"] = data["question_id"]
        return data

    remove_columns = []
    if not keep_all_columns:
        remove_columns = [
            "question_title",
            "contest_id",
            "metadata",
            "platform",
            "question_id",
            "public_test_cases",
            "private_test_cases",
        ]
    dataset = dataset.cast_column("public_test_cases", Value("large_string"))
    dataset = dataset.cast_column("private_test_cases", Value("large_string"))
    dataset = dataset.cast_column("contest_date", Value("string"))
    dataset = dataset.map(map_fn, remove_columns=remove_columns)
    return dataset


def prepare(output_dir, split):
    output_file_path = os.path.join(output_dir, f"{split}.jsonl")

    data = parse_data(split)
    data = clean_data(data)
    print("Len of data: ", len(data))

    print("Writing to file...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file_path, "w") as f:
        for problem in data:
            output_record = {**problem}
            output_record["subset_for_metrics"] = problem["difficulty"]
            output_record["release_version"] = split
            json.dump(output_record, f)
            f.write("\n")


if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    prepare(data_dir, "v5_2408_2501")
    prepare(data_dir, "v6_2408_2505")
