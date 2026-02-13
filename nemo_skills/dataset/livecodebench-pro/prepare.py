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

import json
import os
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download

TESTCASE_REPO = "QAQAQAQAQ/LiveCodeBench-Pro-Testcase"
PROBLEM_REPO = "QAQAQAQAQ/LiveCodeBench-Pro"
DEFAULT_SPLITS = [
    ("24q4", "quater_2024_10_12", 207),
    ("25q1", "quater_2025_1_3", 166),
    ("25q2", "quater_2025_4_6", 167),
    ("25q3", "quater_2025_7_9", 144),
]

FORMATTING_MESSAGE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the c++ program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."


def download_testcases(local_dir, token):
    """
    Downloads the large testcase dataset (~15GB) to the specified directory.
    """
    print(f"Downloading testcases from {TESTCASE_REPO} to {local_dir}...")
    try:
        path = snapshot_download(repo_id=TESTCASE_REPO, repo_type="dataset", local_dir=local_dir, token=token)
        print(f"Testcases successfully downloaded to: {path}")
    except Exception as e:
        print(f"Failed to download testcases: {e}")
        raise


def process_problem_splits(output_dir, token):
    """
    Downloads problem descriptions, converts them to JSONL, and saves them.
    """
    print(f"Processing problem splits from {PROBLEM_REPO}...")

    for tag, split, sample_size in DEFAULT_SPLITS:
        print(f"  - Processing split: {split} -> test_{tag}.jsonl")

        try:
            dataset = load_dataset(PROBLEM_REPO, split=split, token=token)
            if len(dataset) != sample_size:
                print(f"    WARNING: Expected {sample_size} samples for {split}, but got {len(dataset)}.")

            output_file = output_dir / f"test_{tag}.jsonl"

            with open(output_file, "w", encoding="utf-8") as f:
                for row in dataset:
                    output_record = dict(row)
                    output_record["question_content"] = row["problem_statement"]
                    output_record["formatting_message"] = FORMATTING_MESSAGE
                    output_record["starter_code"] = "```cpp\n// YOUR CODE HERE\n```"
                    output_record["subset_for_metrics"] = row["difficulty"]
                    f.write(json.dumps(output_record) + "\n")

        except Exception as e:
            print(f"    Error processing split {split}: {e}")


if __name__ == "__main__":
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable is required.")
        print("Please export it: export HF_TOKEN='hf_...'")
        exit(1)

    data_dir = Path(__file__).absolute().parent
    testcase_dir = data_dir / "testcases"
    download_testcases(local_dir=testcase_dir, token=hf_token)
    process_problem_splits(output_dir=data_dir, token=hf_token)
