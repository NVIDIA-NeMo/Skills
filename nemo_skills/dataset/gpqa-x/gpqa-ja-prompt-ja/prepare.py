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
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from nemo_skills.dataset.utils import get_mcq_fields

"""
Preprocessing adapted from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/gpqa/generative/utils.py
"""


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = text.replace("  ", " ")
    return text


def format_entry(entry):
    choices = [
        preprocess(entry["Incorrect Answer 1"]),
        preprocess(entry["Incorrect Answer 2"]),
        preprocess(entry["Incorrect Answer 3"]),
        preprocess(entry["Correct Answer"]),
    ]
    
    random.shuffle(choices)
    correct_answer_index = choices.index(preprocess(entry["Correct Answer"]))
    return {
        "expected_answer": f"{chr(65 + correct_answer_index)}",
        "explanation": preprocess(entry["Explanation"]),
        "subset_for_metrics": entry["Subdomain"],
        "difficulty": (
            re.split(r'\s*\(', entry["Writer's Difficulty Estimate"])[0]
            if entry["Writer's Difficulty Estimate"] is not None
            else None
        ),
        **get_mcq_fields(entry["Question"], choices),
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

    dataset = load_dataset("Idavidrein/gpqa", f"gpqa_{split}")["train"]
    write_data_to_file(output_file, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="all",
        choices=("all", "extended", "main", "diamond"),
        help="Dataset split to process.",
    )
    parser.add_argument("--random_seed", type=int, default=42)
    args = parser.parse_args()

    if args.split == "all":
        for split in ["extended", "main", "diamond"]:
            save_data(split, args.random_seed)
    else:
        save_data(args.split, args.random_seed)
        