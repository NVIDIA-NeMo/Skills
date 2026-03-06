# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# From https://github.com/YangLabHKUST/UGPhysics/blob/main/codes/utils.py#L126
OB_ANS_TYPE_ID2EN = {
    "IN": "a range interval",
    "TF": "either True or False",
    "EX": "an expression",
    "EQ": "an equation",
    "MC": "one option of a multiple choice question",
    "NV": "a numerical value without units",
    "TUP": "multiple numbers, separated by comma, such as (x, y, z)",
}

SUBSETS = [
    "AtomicPhysics",
    "ClassicalElectromagnetism",
    "ClassicalMechanics",
    "Electrodynamics",
    "GeometricalOptics",
    "QuantumMechanics",
    "Relativity",
    "SemiconductorPhysics",
    "Solid-StatePhysics",
    "StatisticalMechanics",
    "TheoreticalMechanics",
    "Thermodynamics",
    "WaveOptics",
]


def get_prompt_sentence(answer_type, is_multiple_answer):
    """Build the prompt sentence describing the expected answer format.
    Adapted from https://github.com/YangLabHKUST/UGPhysics/blob/main/codes/utils.py#L146
    """
    types = [t.strip() for t in answer_type.split(",")]
    descriptions = [OB_ANS_TYPE_ID2EN.get(t, t) for t in types]
    if not is_multiple_answer:
        return f"The answer of the problem should be {descriptions[0]}."
    elif len(set(descriptions)) == 1:
        return f"The problem has multiple answers, each of them should be {descriptions[0]}."
    else:
        return f"The problem has multiple answers, with the answers in order being {', '.join(descriptions)}."


def get_boxed_answer_example(is_multiple_answer):
    """Get the boxed answer placeholder string for the prompt."""
    if is_multiple_answer:
        return r"\boxed{multiple answers connected with commas}"
    return r"\boxed{answer}(unit)"


def format_entry(entry):
    is_multiple_answer = entry["is_multiple_answer"]
    answer_type = entry["answer_type"]
    return {
        "index": entry["index"],
        "problem": entry["problem"],
        "expected_answer": entry["answers"],
        "solution": entry["solution"],
        "answer_type": answer_type,
        "subset_for_metrics": entry["subject"],
        "language": entry["language"].lower(),
        "is_multiple_answer": is_multiple_answer,
        "prompt_sentence": get_prompt_sentence(answer_type, is_multiple_answer),
        "boxed_answer_example": get_boxed_answer_example(is_multiple_answer),
    }


def load_data(lang_split):
    data = []
    for subset in tqdm(SUBSETS, desc=f"Loading {lang_split} subsets"):
        subset_data = load_dataset("UGPhysics/ugphysics", subset, split=lang_split)
        data.extend(subset_data)
    return data


def save_data(data, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wt", encoding="utf-8") as fout:
        for entry in tqdm(data, desc=f"Writing {output_path.name}"):
            json.dump(format_entry(entry), fout)
            fout.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="all", choices=("all", "en", "zh", "en_zh"))
    args = parser.parse_args()

    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)

    if args.split == "all":
        en_data = load_data("en")
        save_data(en_data, data_dir / "en.jsonl")
        zh_data = load_data("zh")
        save_data(zh_data, data_dir / "zh.jsonl")
        save_data(en_data + zh_data, data_dir / "en_zh.jsonl")
    else:
        if args.split == "en_zh":
            en_data = load_data("en")
            zh_data = load_data("zh")
            data = en_data + zh_data
        else:
            data = load_data(args.split)
        save_data(data, data_dir / f"{args.split}.jsonl")
