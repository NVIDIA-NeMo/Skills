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

import datasets

HF_DATASET = "CohereLabs/m-ArenaHard-v2.0"
SUPPORTED_LANGUAGES = datasets.get_dataset_config_names(HF_DATASET)


def format_entry(row: dict, language: str) -> dict:
    return {
        "question": row["prompt"],
        "question_id": row["question_id"],
        # "hard_prompt" = judge generates its own answer before comparing,
        # "creative_writing" = judge compares directly without generating own answer
        "category": row["category"],
        "subcategory": row.get("subcategory", ""),
        "language": language,
        "subset_for_metrics": language,
    }


def main(args):
    invalid_langs = set(args.languages) - set(SUPPORTED_LANGUAGES)
    if invalid_langs:
        raise ValueError(f"Unsupported languages: {invalid_langs}. Supported: {SUPPORTED_LANGUAGES}")

    data_dir = Path(__file__).absolute().parent
    output_file = data_dir / "test.jsonl"

    all_entries = []
    for language in args.languages:
        print(f"Processing {language}...")
        ds = datasets.load_dataset(HF_DATASET, name=language, split="test")
        for row in ds:
            all_entries.append(format_entry(row, language))

    # Populate baseline_answer from a pre-generated JSONL file (e.g. output of generate pipeline)
    if args.baseline_file:
        baseline_lookup = {}
        with open(args.baseline_file, "rt", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                lang, qid = data["language"], data["question_id"]
                baseline_lookup[(lang, qid)] = data["generation"]
        for entry in all_entries:
            lang, qid = entry["language"], entry["question_id"]
            if (lang, qid) not in baseline_lookup:
                raise ValueError(f"({lang}, {qid}) not found in baseline file")
            entry["baseline_answer"] = baseline_lookup[(lang, qid)]

    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in all_entries:
            json.dump(entry, fout, ensure_ascii=False)
            fout.write("\n")
    print(f"Saved {len(all_entries)} entries to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare m-ArenaHard-v2.0 multilingual benchmark.")
    parser.add_argument(
        "--languages",
        default=SUPPORTED_LANGUAGES,
        nargs="+",
        help=f"Languages to include. Supported: {SUPPORTED_LANGUAGES}",
    )
    parser.add_argument(
        "--baseline-file",
        default=None,
        help="Path to JSONL with baseline answers (must have 'question_id' and 'generation').",
    )
    args = parser.parse_args()
    main(args)
