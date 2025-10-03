# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import random
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from nemo_skills.dataset.utils import get_mcq_fields

"""
Preprocessing adapted from nemo_skills/dataset/gpqa/prepare.py
"""


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace("  ", " ")
    return text


def format_entry(entry):
    uuid: str = entry["uuid"]
    question: str = entry["question"]
    options: list[str] = entry["options"]
    answer_letter: str = entry["answer_letter"]
    answer: str = entry["answer"]
    discipline: str = entry["discipline"]
    field: str = entry["field"]
    subfield: str = entry["subfield"]
    difficulty: str = entry["difficulty"]
    is_calculation: bool = entry["is_calculation"]

    assert ord(answer_letter) >= ord("A") and ord(answer_letter) <= ord("J")
    answer_index = ord(answer_letter) - ord("A")

    choices = [preprocess(option) for option in options]
    if len(choices) != len(set(choices)):
        raise ValueError(f"Choices are not unique: {choices}")

    correct_answer = choices[answer_index]
    random.shuffle(choices)
    correct_answer_index = choices.index(correct_answer)
    assert correct_answer_index >= 0 and correct_answer_index < len(choices)
    assert preprocess(answer) == choices[correct_answer_index]  # recheck after shuffling

    return {
        "expected_answer": f"{chr(ord('A') + correct_answer_index)}",
        "uuid": uuid,
        "subset_for_metrics": entry["discipline"],
        "discipline": discipline,
        "field": field,
        "subfield": subfield,
        "difficulty": difficulty,
        "is_calculation": is_calculation,
        **get_mcq_fields(question, choices),
    }


def write_data_to_file(output_file, data):
    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in tqdm(data, desc=f"Writing {output_file.name}"):
            json.dump(format_entry(entry), fout)
            fout.write("\n")


def save_data(split, random_seed):
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)

    output_file = data_dir / f"{split}.jsonl"
    random.seed(random_seed)
    dataset = load_dataset("m-a-p/SuperGPQA")["train"]
    if split == "science":
        dataset = dataset.filter(lambda x: x["discipline"] == "Science")
    elif split == "test":
        dataset = dataset
    else:
        raise ValueError(f"Invalid split: {split}")

    write_data_to_file(output_file, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="all",
        choices=("all", "test", "science"),
        help="Dataset split to process.",
    )
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    if args.split == "all":
        for split in ["test", "science"]:
            save_data(split, args.random_seed)
    else:
        save_data(args.split, args.random_seed)
