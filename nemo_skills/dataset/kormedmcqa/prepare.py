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

answer_map = {
    1: "A",
    2: "B",
    3: "C",
    4: "D",
    5: "E",
}

# Matches the '정답: X' answer line the prompt instructs the model to produce.
_EXTRACT_REGEX = r"(?i)정답\s*[:：]\s*\**\(?([A-E])\)?\**"


def format_entry(entry):
    answer = answer_map[entry["answer"]]
    subject = entry["subject"]
    options = [entry["A"], entry["B"], entry["C"], entry["D"], entry["E"]]

    return {
        "expected_answer": answer,
        "extract_from_boxed": False,
        "extract_regex": _EXTRACT_REGEX,
        "relaxed": False,
        "subset_for_metrics": subject,
        **get_mcq_fields(entry["question"], options),
    }


def write_data_to_file(output_file, data):
    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in tqdm(data, desc=f"Writing {output_file.name}"):
            json.dump(format_entry(entry), fout, ensure_ascii=False)
            fout.write("\n")


def main(args):
    dataset = load_dataset(
        "sean0042/KorMedMCQA", data_files={args.split: f"*/{args.split}-*.parquet"}, split=args.split
    )
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / f"{args.split}.jsonl"
    write_data_to_file(output_file, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("train", "dev", "test"), help="Dataset split to process.")
    args = parser.parse_args()
    main(args)
