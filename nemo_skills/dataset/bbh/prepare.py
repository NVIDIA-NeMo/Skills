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
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# The 27 BIG-Bench Hard tasks (Suzgun et al., 2022 - https://arxiv.org/abs/2210.09261).
TASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]

# Multiple-choice tasks encode the answer as a parenthesized letter, e.g. "(A)".
# Strip the parentheses to the bare letter so the math grader's multiple-choice
# path matches robustly whether the model boxes "A" or "(A)".
MCQ_TARGET = re.compile(r"^\(([A-Z])\)$")


def format_entry(entry: dict, task: str) -> dict:
    target = str(entry["target"]).strip()
    match = MCQ_TARGET.match(target)
    expected_answer = match.group(1) if match else target
    return {
        "problem": entry["input"],
        "expected_answer": expected_answer,
        "subset_for_metrics": task,
        "subtopic": task,
    }


def save_data(split: str):
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / f"{split}.jsonl"

    with open(output_file, "wt", encoding="utf-8") as fout:
        for task in tqdm(TASKS, desc="Preparing BBH tasks"):
            dataset = load_dataset("lukaemon/bbh", task)[split]
            for entry in dataset:
                fout.write(json.dumps(format_entry(entry, task)) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="test",
        choices=("test",),
        help="BBH only provides a test split.",
    )
    args = parser.parse_args()

    save_data(args.split)
