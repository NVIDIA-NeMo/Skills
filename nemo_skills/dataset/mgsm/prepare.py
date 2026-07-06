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

import argparse
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

# MGSM (Shi et al., 2022 - https://arxiv.org/abs/2210.03057): 250 GSM8K test
# problems translated into 10 languages, plus the original English. Language codes
# are ISO 639-1 (compatible with the "math_multilingual" metric's target_language).
LANGUAGES = ["en", "es", "fr", "de", "ru", "zh", "ja", "th", "sw", "bn", "te"]


def format_entry(entry: dict, lang: str) -> dict:
    return {
        "problem": entry["question"],
        "expected_answer": entry["answer_number"],
        "subset_for_metrics": lang,
        "target_language": lang,
    }


def save_data(split: str, languages: list[str]):
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / f"{split}.jsonl"

    with open(output_file, "wt", encoding="utf-8") as fout:
        for lang in tqdm(languages, desc="Preparing MGSM languages"):
            dataset = load_dataset("juletxara/mgsm", lang)[split]
            for entry in dataset:
                # Non-latin scripts must be preserved, so keep ensure_ascii=False.
                fout.write(json.dumps(format_entry(entry, lang), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--languages",
        default=LANGUAGES,
        nargs="+",
        choices=LANGUAGES,
        help="Which language(s) to prepare. Defaults to all 11 MGSM languages.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("test", "train"),
        help="MGSM provides a test split (250 problems/language) and a small train split of few-shot exemplars.",
    )
    args = parser.parse_args()

    save_data(args.split, args.languages)
