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
import os.path
from collections import defaultdict
from pathlib import Path

import requests
from datasets import load_dataset

run_url = "https://raw.githubusercontent.com/huggingface/ioi/refs/heads/main/run_tests/custom_setup/run"
compile_url = "https://raw.githubusercontent.com/huggingface/ioi/refs/heads/main/run_tests/custom_setup/compile"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--suffix", type=str, default="24")
    args = parser.parse_args()

    data_dir = Path(__file__).absolute().parent

    run_code = requests.get(run_url).text
    compile_code = requests.get(compile_url).text

    ds = load_dataset("open-r1/ioi", split=args.split)
    entries = []
    for x, item in enumerate(ds):
        # remove the examples for the test split, as we do not need to compute evaluation for them.
        if item["score"] != 0 and args.split == "test":
            entries.append(
                {
                    "id": x,
                    "name": item["name"],
                    "ioi_id": item["id"],
                    "subtask": item["subtask"],
                    "question": item["problem"],
                    "subtask_score": item["score"],
                }
            )

    with open(os.path.join(data_dir, f"ioi{args.suffix}.jsonl"), "w") as f:
        f.write("\n".join(json.dumps(x) for x in entries))

    tests_dataset = load_dataset("open-r1/ioi-test-cases", name="2024", split="train")

    # First, parse the tests_dataset to build a mapping:
    #   problem_id -> test_name -> {input, output}
    test_cases = defaultdict(dict)
    for test_case in tests_dataset:
        tname = test_case["test_name"]
        problem_id = test_case["problem_name"]
        test_cases[problem_id][tname] = {"input": test_case["test_input"], "output": test_case["test_output"]}

    final_structure = defaultdict(lambda: defaultdict(dict))
    for entry in ds:
        problem_id = entry["name"]
        subtask = entry["subtask"]
        if subtask == "00-samples":  # skip the sample subtasks.
            continue
        test_names = entry["test_names"]
        tests = {}
        for test_name in test_names:
            tests[test_name] = test_cases[problem_id][test_name]
        final_structure[problem_id][subtask] = {
            "tests": tests,
            "subtask_score": entry["score"],
            "score_precision": entry["score_precision"],
            "run": run_code,
            "compile": compile_code,
            "grader_files": entry["grader_files"],
        }

    with open(os.path.join(data_dir, f"ioi{args.suffix}_metadata.json"), "w") as f:
        json.dump(final_structure, f)
