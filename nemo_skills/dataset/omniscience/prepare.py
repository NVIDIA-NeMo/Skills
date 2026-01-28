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

TOPIC_TO_SPLIT_MAP = {
    "Humanities and Social Sciences": "humanities",
    "Health": "health",
    "Software Engineering": "swe",
    "Science Engineering and Mathematics": "stem",
    "Law": "law",
    "Finance": "finance",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--splits",
        default=["text", "humanities", "health", "swe", "stem", "law", "finance"],
        nargs="+",
        choices=["text", "humanities", "health", "swe", "stem", "law", "finance"],
    )
    return parser.parse_args()


def format_entry(entry) -> dict:
    return {
        "id": entry["question_id"],
        "domain": entry["domain"],
        "topic": entry["topic"],
        "question": entry["question"],
        "expected_answer": entry["answer"],
    }


def write_jsonl(data: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for d in data:
            f.write(json.dumps(d) + "\n")


if __name__ == "__main__":
    args = parse_args()

    dataset = load_dataset("ArtificialAnalysis/AA-Omniscience-Public", split="train")
    jsonl_data = [format_entry(d) for d in dataset]
    output_dir = Path(__file__).absolute().parent

    split_set = set(args.splits)
    splits = {
        "text": dataset,
        **{
            TOPIC_TO_SPLIT_MAP.get(t, str(t).lower()): dataset.filter(lambda x: x["domain"] == t)
            for t in dataset.unique("domain")
        },
    }
    splits = {k: v for k, v in splits.items() if k in split_set}

    for split, data in tqdm(splits.items(), total=len(splits)):
        output_file = output_dir / f"{split}.jsonl"
        formatted_data = [format_entry(entry) for entry in data]
        write_jsonl(formatted_data, output_file)
