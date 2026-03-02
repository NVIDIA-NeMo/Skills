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
import random
from pathlib import Path


def split_generation_field(
    input_folder: str, train_out: str, test_out: str, test_size: float = 0.2, seed: int = 42, max_examples=10000
):
    train_path = Path(train_out)
    test_path = Path(test_out)

    incidents = []
    with open(os.path.join(input_folder, "iteration_0.jsonl"), "r", encoding="utf-8") as f_in:
        for line in f_in:
            row = json.loads(line)
            number = row.get("incident_identifier", row.get("number"))
            incidents.append(number)

    random.seed(seed)
    random.shuffle(incidents)

    incidents = incidents[:max_examples]
    n = len(incidents)
    n_test = max(1, int(round(n * test_size))) if n > 0 else 0
    n_train = n - n_test
    train_set = set(incidents[:n_train])
    test_set = set(incidents[n_train:])

    train_rows = []
    test_rows = []

    i = 0
    while os.path.exists(os.path.join(input_folder, f"iteration_{i}.jsonl")):
        current_iteration_train = []
        current_iteration_test = []
        with open(os.path.join(input_folder, f"iteration_{i}.jsonl"), "r", encoding="utf-8") as f_in:
            for line in f_in:
                row = json.loads(line)
                number = row.get("incident_identifier", row.get("number"))
                if number in train_set:
                    current_iteration_train.append(row)

        random.shuffle(current_iteration_train)
        train_rows += current_iteration_train
        i += 1

    i = 0
    while os.path.exists(os.path.join(input_folder, f"iteration_{i}.jsonl")):
        current_iteration_train = []
        current_iteration_test = []
        with open(os.path.join(input_folder, f"iteration_{i}.jsonl"), "r", encoding="utf-8") as f_in:
            for line in f_in:
                row = json.loads(line)
                number = row.get("incident_identifier", row.get("number"))
                if number in test_set:
                    resolution = row.get("root_cause_secondary") or row.get("close_code", "")
                    if resolution:
                        row["expected"] = f"Close Code: [{resolution}]"
                        row["initial_background"] = row["background"]
                        row["background"] = "\n<think>\n"
                        current_iteration_test.append(row)

        random.shuffle(current_iteration_test)
        test_rows += current_iteration_test
        i += 1

    train_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.parent.mkdir(parents=True, exist_ok=True)

    with open(train_path, "w", encoding="utf-8") as f_train:
        for r in train_rows:
            f_train.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(test_path, "w", encoding="utf-8") as f_test:
        for r in test_rows:
            f_test.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Total lines read: {n}")
    print(f"Train size: {len(train_set)}  |  Test size: {len(test_set)}")
    print(f"Train path: {train_path}")
    print(f"Test path:  {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split JSONL incidents with 'generation' field into training data")
    parser.add_argument("--input_dir", help="Path to input_dir JSONL file", default="output_incident.jsonl")
    parser.add_argument("--train_output", help="Path to output JSONL file", default="training_data_split.jsonl")
    parser.add_argument("--test_output", help="Path to output JSONL file", default="testing_data_split.jsonl")
    parser.add_argument(
        "--preview", type=int, default=2, help="Number of examples to preview before confirmation (default: 2)"
    )
    args = parser.parse_args()

    split_generation_field(args.input_dir, args.train_output, args.test_output)
