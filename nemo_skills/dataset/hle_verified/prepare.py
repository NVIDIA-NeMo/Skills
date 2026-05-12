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

HLE_CATEGORIES_MAP = {
    "Other": "other",
    "Humanities/Social Science": "human",
    "Math": "math",
    "Physics": "phy",
    "Computer Science/AI": "cs",
    "Biology/Medicine": "bio",
    "Chemistry": "chem",
    "Engineering": "eng",
}

# Reverse mapping for filtering
HLE_REVERSE_MAP = {v: k for k, v in HLE_CATEGORIES_MAP.items()}

HLE_VERIFIED_CLASSES_MAP = {
    "Gold subset": "gold",
    "Revision subset": "revision",
    "Uncertain subset": "uncertain",
}

# Reverse mapping for filtering
HLE_VERIFIED_CLASSES_REVERSE_MAP = {v: k for k, v in HLE_VERIFIED_CLASSES_MAP.items()}

REPO_ID = "skylenage/HLE-Verified"


def load_dataset_from_hub():
    """Load the dataset from HuggingFace hub.

    Fields not exposed as top-level columns (author_name, rationale, answer_type,
    canary, image) are stored as a JSON string in the 'json' column and parsed here.
    """
    df = load_dataset(REPO_ID, split="train").to_pandas()

    parsed = df["json"].apply(json.loads)
    for field in ("author_name", "rationale", "answer_type", "canary", "image"):
        df[field] = parsed.apply(lambda x, f=field: x.get(f))

    return df


def format_entry(entry):
    return {
        "id": entry["id"],
        "problem": entry["question"],
        "expected_answer": entry["answer"],
        "answer_type": entry["answer_type"],
        "reference_solution": entry["rationale"],
        "raw_subject": entry["raw_subject"],
        "subset_for_metrics": entry["category"],
        "author_name": entry["author_name"],
        "canary": entry["canary"],
        "verified_class": HLE_VERIFIED_CLASSES_MAP.get(entry["Verified_Classes"], entry["Verified_Classes"]),
    }


def write_data_to_file(output_file, data, split):
    with open(output_file, "wt", encoding="utf-8") as fout:
        for _, entry in tqdm(data.iterrows(), total=len(data), desc=f"Writing {output_file.name}"):
            # Filter by category for category-specific splits
            if split in HLE_REVERSE_MAP and entry["category"] != HLE_REVERSE_MAP[split]:
                continue
            # Filter by verified class for class-specific splits
            if split in HLE_VERIFIED_CLASSES_REVERSE_MAP:
                if entry["Verified_Classes"] != HLE_VERIFIED_CLASSES_REVERSE_MAP[split]:
                    continue
            if entry["image"]:
                continue
            # text split = text-only entries from Gold + Revision subsets only
            if split == "text" and entry["Verified_Classes"] == HLE_VERIFIED_CLASSES_REVERSE_MAP["uncertain"]:
                continue
            json.dump(format_entry(entry), fout)
            fout.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="all",
        choices=("all", "text") + tuple(HLE_CATEGORIES_MAP.values()) + tuple(HLE_VERIFIED_CLASSES_MAP.values()),
        help="Dataset split to process (all/text/math/other/human/phy/cs/bio/chem/eng/gold/revision/uncertain).",
    )
    args = parser.parse_args()
    dataset = load_dataset_from_hub()
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    if args.split == "all":
        all_splits = ["text"] + list(HLE_CATEGORIES_MAP.values()) + list(HLE_VERIFIED_CLASSES_MAP.values())
        for split in all_splits:
            output_file = data_dir / f"{split}.jsonl"
            write_data_to_file(output_file, dataset, split)
    else:
        output_file = data_dir / f"{args.split}.jsonl"
        write_data_to_file(output_file, dataset, args.split)
