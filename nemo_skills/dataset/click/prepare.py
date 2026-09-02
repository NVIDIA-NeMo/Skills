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
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from nemo_skills.dataset.utils import get_mcq_fields

# Matches the '정답: X' answer line the prompt instructs the model to produce.
# CLIcK mixes 4-choice and 5-choice items; the 5-choice prompt covers both.
_EXTRACT_REGEX = r"(?i)정답\s*[:：]\s*\**\(?([A-E])\)?\**"


def format_entry(entry):
    # The schema stores the answer as the literal text of the correct
    # choice; map it back to the corresponding letter so the per-sample
    # regex can match.
    answer_index = entry["choices"].index(entry["answer"])
    expected_letter = chr(ord("A") + answer_index)

    # Some items carry an extra `paragraph` context that the question
    # refers to; prepend it when present.
    paragraph = entry["paragraph"].strip()
    question = entry["question"].strip()
    if paragraph:
        question = f"{paragraph}\n\n{question}"

    return {
        "expected_answer": expected_letter,
        "extract_from_boxed": False,
        "extract_regex": _EXTRACT_REGEX,
        "relaxed": False,
        "subset_for_metrics": entry["subcategory"],
        "id": entry["id"],
        **get_mcq_fields(question, entry["choices"]),
    }


def write_data_to_file(output_file, data):
    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in tqdm(data, desc=f"Writing {output_file.name}"):
            json.dump(format_entry(entry), fout, ensure_ascii=False)
            fout.write("\n")


def main(args):
    # bzantium/CLIcK is a mirror of the upstream CLIcK dataset
    # (github.com/rladmstn1714/CLIcK, LREC-COLING 2024) with the per-item
    # subcategory and broad-category labels that the EunsuKim/CLIcK HF mirror
    # drops. See the dataset card for details.
    dataset = load_dataset("bzantium/CLIcK", split="test")
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / f"{args.split}.jsonl"
    write_data_to_file(output_file, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("test",), help="Dataset split to process.")
    args = parser.parse_args()
    main(args)
