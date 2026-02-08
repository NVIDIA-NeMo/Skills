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

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from datasets import Value, load_dataset
from dateutil.relativedelta import relativedelta


class PromptConstants:
    # reference: https://github.com/LiveCodeBench/LiveCodeBench/blob/main/lcb_runner/prompts/code_generation.py#L35C5-L38C1
    FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."


def parse_data(release_version="release_latest"):
    data = load_dataset(
        "livecodebench/code_generation_lite", split="test", version_tag=release_version, trust_remote_code=True
    )
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


def get_first_last_day(year_month_str):
    try:
        date_obj = datetime.strptime(year_month_str, "%Y-%m")
        first_day = date_obj.date().replace(day=1)
        last_day = (date_obj + relativedelta(months=1, days=-1)).date()
        return first_day, last_day
    except ValueError:
        raise ValueError("Invalid date format. Please use '%Y-%m'.")


def parse_month_range(start_date, end_date):
    try:
        start_date, _ = get_first_last_day(start_date)
        _, end_date = get_first_last_day(end_date)
        return start_date, end_date
    except ValueError as e:
        raise ValueError(str(e))


def clean_data(dataset, keep_all_columns=False):
    def map_fn(data):
        if data["starter_code"]:
            data["formatting_message"] = PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE
            data["starter_code"] = f"```python\n{data['starter_code']}\n```"
        else:
            data["formatting_message"] = PromptConstants.FORMATTING_WITHOUT_STARTER_CODE
            data["starter_code"] = "```python\n# YOUR CODE HERE\n```"

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
    dataset = dataset.map(map_fn, remove_columns=remove_columns)
    return dataset


def prepare(start_date, end_date, release_version, output_dir, keep_all_columns=False):
    start_date, end_date = parse_month_range(start_date, end_date)
    start_yymm = start_date.strftime("%y%m")
    end_yymm = end_date.strftime("%y%m")
    output_file_path = os.path.join(output_dir, f"test_{release_version}_{start_yymm}_{end_yymm}.jsonl")

    assert release_version in ["v1", "v2", "v3", "v4", "v5", "v6"]

    data = parse_data(release_version=f"release_{release_version}")
    data = clean_data(data, keep_all_columns)
    print("Len of data: ", len(data))

    print("Writing to file...")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file_path, "w") as f:
        for problem in data:
            input_date = datetime.strptime(problem["contest_date"], "%Y-%m-%dT%H:%M:%S").date()
            if start_date <= input_date <= end_date:
                output_record = {**problem}
                output_record["subset_for_metrics"] = problem["difficulty"]
                output_record["release_version"] = release_version
                json.dump(output_record, f)
                f.write("\n")


DEFAULT_SPLITS = [
    ("v5", "2024-08", "2025-02"),  # previous default
    ("v5", "2024-07", "2024-12"),  # aai split
    ("v5", "2024-10", "2025-02"),
    ("v6", "2024-08", "2025-05"),  # current default in lb
]

if __name__ == "__main__":
    # Write an argparse to a json file, read it in and parse it
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default=str(Path(__file__).parent))
    parser.add_argument("--release_version", type=str, default="all")
    parser.add_argument("--start_date", type=str, default="all", help="End date in YYYY-MM format")
    parser.add_argument("--end_date", type=str, default="all", help="End date in YYYY-MM format")
    parser.add_argument("--keep_all_columns", action="store_true", help="keep all columns in the output jsonl file")

    args = parser.parse_args()

    if args.release_version == "all" and args.start_date == "all" and args.end_date == "all":
        # Prepare all splits
        for release_version, start_date, end_date in DEFAULT_SPLITS:
            print(f"Processing data for {release_version} from {start_date} to {end_date}")
            prepare(start_date, end_date, release_version, args.output_dir, args.keep_all_columns)
    else:
        if args.release_version == "all" or args.start_date == "all" or args.end_date == "all":
            raise ValueError(
                "If preparing a custom split, you must specify all "
                "--release_version, --start_date, and --end_date arguments."
            )
        prepare(args.start_date, args.end_date, args.release_version, args.output_dir, args.keep_all_columns)

    # test_v5_2408_2502.jsonl: 279 samples
    # test_v5_2410_2502.jsonl: 166 samples
    # test_v5_2410_2504.jsonl: 166 samples
    # test_v6_2408_2502.jsonl: 374 samples
    # test_v6_2410_2502.jsonl: 261 samples
    # test_v6_2410_2504.jsonl: 341 samples
