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

# HUMSS / STEM / Applied Science / Other groupings come from the KMMLU paper
# (Son et al. 2024) and aren't exposed via the HF dataset metadata.
supercategories = {
    "Accounting": "HUMSS",
    "Agricultural-Sciences": "Other",
    "Aviation-Engineering-and-Maintenance": "Applied Science",
    "Biology": "STEM",
    "Chemical-Engineering": "STEM",
    "Chemistry": "STEM",
    "Civil-Engineering": "STEM",
    "Computer-Science": "STEM",
    "Construction": "Other",
    "Criminal-Law": "HUMSS",
    "Ecology": "STEM",
    "Economics": "HUMSS",
    "Education": "HUMSS",
    "Electrical-Engineering": "STEM",
    "Electronics-Engineering": "Applied Science",
    "Energy-Management": "Applied Science",
    "Environmental-Science": "Applied Science",
    "Fashion": "Other",
    "Food-Processing": "Other",
    "Gas-Technology-and-Engineering": "Applied Science",
    "Geomatics": "Applied Science",
    "Health": "Other",
    "Industrial-Engineer": "Applied Science",
    "Information-Technology": "STEM",
    "Interior-Architecture-and-Design": "Other",
    "Law": "HUMSS",
    "Machine-Design-and-Manufacturing": "Applied Science",
    "Management": "HUMSS",
    "Maritime-Engineering": "Applied Science",
    "Marketing": "Other",
    "Materials-Engineering": "STEM",
    "Mechanical-Engineering": "STEM",
    "Nondestructive-Testing": "Applied Science",
    "Patent": "Other",
    "Political-Science-and-Sociology": "HUMSS",
    "Psychology": "HUMSS",
    "Public-Safety": "Other",
    "Railway-and-Automotive-Engineering": "Applied Science",
    "Real-Estate": "Other",
    "Refrigerating-Machinery": "Other",
    "Social-Welfare": "HUMSS",
    "Taxation": "HUMSS",
    "Telecommunications-and-Wireless-Technology": "Applied Science",
    "Korean-History": "HUMSS",
    "Math": "STEM",
}

answer_map = {
    1: "A",
    2: "B",
    3: "C",
    4: "D",
}

# Matches the '정답: X' answer line the prompt instructs the model to produce.
_EXTRACT_REGEX = r"(?i)정답\s*[:：]\s*\**\(?([A-D])\)?\**"


def format_entry(entry):
    answer = answer_map[entry["answer"]]
    category = entry["Category"]
    options = [entry["A"], entry["B"], entry["C"], entry["D"]]

    return {
        "expected_answer": answer,
        "extract_from_boxed": False,
        "extract_regex": _EXTRACT_REGEX,
        "relaxed": False,
        "subtopic": category,
        "subset_for_metrics": supercategories[category.replace(" ", "-")],
        **get_mcq_fields(entry["question"], options),
    }


def write_data_to_file(output_file, data):
    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in tqdm(data, desc=f"Writing {output_file.name}"):
            json.dump(format_entry(entry), fout, ensure_ascii=False)
            fout.write("\n")


def main(args):
    dataset = load_dataset("HAERAE-HUB/KMMLU", data_files={args.split: f"data/*-{args.split}.csv"}, split=args.split)
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / f"{args.split}.jsonl"
    write_data_to_file(output_file, dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("test",), help="Dataset split to process.")
    args = parser.parse_args()
    main(args)
