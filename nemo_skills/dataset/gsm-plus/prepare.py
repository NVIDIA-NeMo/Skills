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
import pathlib
from pathlib import Path

from datasets import load_dataset

from nemo_skills.dataset.utils import add_rounding_instruction

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--categories",
        nargs="+",
        default=[
            "adding_operation",
            "critical_thinking",
            "digit_expansion",
            "distraction_insertion",
            "integer-decimal-fraction_conversion",
            "numerical_substitution",
            "problem_understanding",
            "reversing_operation",
        ],
    )
    parser.add_argument("--no_rounding_instructions", action="store_true")
    parser.add_argument("--cleaning", choices=["none", "light", "hard"], default="none")
    args = parser.parse_args()

    split = "test"
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    output_file = str(data_dir / f"{split}.jsonl")

    dataset = load_dataset("qintongli/GSM-Plus", split="test")

    file_rounded = None
    if not args.no_rounding_instructions:
        output_file_rounded = str(data_dir / f"{split}_rounded.jsonl")
        file_rounded = open(output_file_rounded, "w")

    with open(os.path.join(pathlib.Path(__file__).parent, "cleaned_indexes.json")) as f:
        cleaning_options = json.load(f)
        for key in cleaning_options.keys():
            cleaning_options[key] = set(cleaning_options[key])

    cleaning_options["none"] = set(range(len(dataset)))
    with open(output_file, "w") as test_full:
        for i, original_entry in enumerate(dataset):
            if (
                original_entry["perturbation_type"].replace(" ", "_") in args.categories
                and i in cleaning_options[args.cleaning]
            ):
                # original entries
                reference_solution = original_entry.get("solution", None) or original_entry.get(
                    "reference_solution", None
                )
                expected_answer = original_entry.get("answer", None) or original_entry.get("expected_answer", None)
                expected_answer = expected_answer if expected_answer != "None" else "insufficient"
                entry = dict(
                    problem=original_entry["question"],
                    reference_solution=reference_solution,
                    expected_answer=expected_answer,
                    **{
                        key: value
                        for key, value in original_entry.items()
                        if key
                        not in [
                            "answer",
                            "expected_answer",
                            "solution",
                            "problem",
                            "reference_solution",
                        ]
                    },
                )
                # converting to int if able to for cleaner text representation
                if str(entry["expected_answer"]).replace(".", "", 1).replace("-", "", 1).isdigit():
                    entry["expected_answer"] = float(entry["expected_answer"])
                    if int(entry["expected_answer"]) == entry["expected_answer"]:
                        entry["expected_answer"] = int(entry["expected_answer"])

                test_full.write(json.dumps(entry) + "\n")

                if file_rounded:
                    entry_rounded = add_rounding_instruction(entry)
                    file_rounded.write(json.dumps(entry_rounded) + "\n")

    if file_rounded:
        file_rounded.close()
