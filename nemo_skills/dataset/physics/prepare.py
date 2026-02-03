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

import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def strip_boxed(s):
    """Remove \\boxed{} if present"""
    if s.startswith("\\boxed{") and s.endswith("}"):
        return s[7:-1]
    return s


def process_answer(answer):
    """Flatten all answers and wrap in a single \\boxed{}"""
    all_answers = [strip_boxed(item) for sublist in answer for item in sublist]
    return f"\\boxed{{{', '.join(all_answers)}}}"


def format_entry(entry):
    return {
        "problem": entry["question"],
        "expected_answer": process_answer(entry["answer"]),
        "solution": entry["solution"],
        "answer_type": entry["answer_type"],
        "subset_for_metrics": entry["domain"],
        "difficulty": entry["difficulty"],
        "language": entry["language"],
    }


def write_data_to_file(output_file, data):
    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in tqdm(data, desc=f"Writing {output_file.name}"):
            json.dump(format_entry(entry), fout)
            fout.write("\n")


def save_data(split_data, split_name):
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / f"{split_name}.jsonl"

    write_data_to_file(output_file, split_data)


if __name__ == "__main__":
    dataset = load_dataset("desimfj/PHYSICS")["test"]
    eng_data = [entry for entry in dataset if entry["language"] == "en"]
    ch_data = [entry for entry in dataset if entry["language"] == "zh"]
    full_data = eng_data + ch_data

    for split_data, split_name in zip([eng_data, ch_data, full_data], ["test", "zh", "en_zh"]):
        save_data(split_data, split_name)
