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
import re
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from nemo_skills.dataset.utils import get_mcq_fields

# Matches the '정답: X' answer line the prompt instructs the model to produce.
_EXTRACT_REGEX = r"(?i)정답\s*[:：]\s*\**\(?([A-J])\)?\**"

# KoBALT-700 ships each question as a single 'Question' string with the 10
# options embedded as `A: ...`, `B: ...`, ... `J: ...` lines. We split the
# stem from the options here so the prompt template can render them in the
# same format as the other Korean MCQ benchmarks.
_OPTION_SPLIT = re.compile(r"\n(?=[A-J][:.\)]\s)")
_OPTION_PREFIX = re.compile(r"^[A-J][:.\)]\s+")


def _split_question(question_text: str) -> tuple[str, list[str]]:
    first_option = re.search(r"\nA[:.\)]\s", question_text)
    stem = question_text[: first_option.start()].rstrip("\n")
    options_block = question_text[first_option.start() + 1 :]
    options = [_OPTION_PREFIX.sub("", o).strip() for o in _OPTION_SPLIT.split(options_block)]
    return stem, options


def format_entry(entry):
    stem, options = _split_question(entry["Question"])
    return {
        "expected_answer": entry["Answer"],
        "extract_from_boxed": False,
        "extract_regex": _EXTRACT_REGEX,
        "relaxed": False,
        "subset_for_metrics": entry["Class"],
        **get_mcq_fields(stem, options),
    }


def write_data_to_file(output_file, data):
    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in tqdm(data, desc=f"Writing {output_file.name}"):
            json.dump(format_entry(entry), fout, ensure_ascii=False)
            fout.write("\n")


def main(args):
    dataset = load_dataset("snunlp/KoBALT-700", "kobalt_v1", split="raw")
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / f"{args.split}.jsonl"
    write_data_to_file(output_file, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("test",), help="Dataset split to process.")
    args = parser.parse_args()
    main(args)
