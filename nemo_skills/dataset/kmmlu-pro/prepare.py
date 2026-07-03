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
    "1": "A",
    "2": "B",
    "3": "C",
    "4": "D",
    "5": "E",
}

# Matches the '정답: X' answer line the prompt instructs the model to produce.
_EXTRACT_REGEX = r"(?i)정답\s*[:：]\s*\**\(?([A-E])\)?\**"

# Surface the per-license breakdown in English so non-Korean-speaking
# readers of the metrics output can interpret it. Names match the
# Korean National Professional Licensure (KNPL) labels reported in the
# KMMLU-Pro paper (arXiv:2507.08924, Table "Names of KNPLs (EN)").
_LICENSE_EN = {
    "감정평가사": "Certified Appraiser",
    "공인노무사": "Certified Public Labor Attorney",
    "공인회계사": "Certified Public Accountant",
    "관세사": "Certified Customs Broker",
    "법무사": "Certified Judicial Scrivener",
    "변리사": "Certified Patent Attorney",
    "변호사": "Lawyer",
    "세무사": "Certified Tax Accountant",
    "손해사정사": "Certified Damage Adjuster",
    "약사": "Pharmacist",
    "의사": "Physician",
    "치과의사": "Dentist",
    "한약사": "Herb Pharmacist",
    "한의사": "Doctor of Korean Medicine",
}


def format_entry(entry):
    return {
        "expected_answer": answer_map[entry["solution"]],
        "extract_from_boxed": False,
        "extract_regex": _EXTRACT_REGEX,
        "relaxed": False,
        "subset_for_metrics": _LICENSE_EN.get(entry["license_name"], entry["license_name"]),
        **get_mcq_fields(entry["question"], entry["options"]),
    }


def write_data_to_file(output_file, data):
    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in tqdm(data, desc=f"Writing {output_file.name}"):
            json.dump(format_entry(entry), fout, ensure_ascii=False)
            fout.write("\n")


def main(args):
    dataset = load_dataset("LGAI-EXAONE/KMMLU-Pro")[args.split]
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / f"{args.split}.jsonl"
    write_data_to_file(output_file, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("test",), help="Dataset split to process.")
    args = parser.parse_args()
    main(args)
