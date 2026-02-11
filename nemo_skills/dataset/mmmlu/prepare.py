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
from nemo_skills.dataset.mmmlu.mmmlu_utils import (
    download_mmmlu_datasets,
    SUPPORTED_LANGUAGES,
    Schema,
    subject2category,
    get_mcq_fields,
    MULTILINGUAL_ANSWER_PATTERN_TEMPLATE,
    MULTILINGUAL_ANSWER_REGEXES,
)


def format_entry(entry: dict, language: str) -> dict:
    expected_answer = entry[Schema.ANSWER]
    category = subject2category.get(entry[Schema.SUBJECT], "other")
    regexes = [
        MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(answer_regex)
        for answer_regex in MULTILINGUAL_ANSWER_REGEXES
    ]
    LETTER_REGEX = r"\b\(?\s*([A-D]|[أ-د]|[অ]|[ব]|[ড]|[ঢ]|[Ａ]|[Ｂ]|[Ｃ]|[Ｄ])\s*\)?\.?\b"
    GREEDY_REGEX = r"[\s\S]*" + LETTER_REGEX
    regexes.append(GREEDY_REGEX) # Matches the last A/B/C/D letter in the response
    return {
        "expected_answer": expected_answer,
        "extract_from_boxed": False,
        "extract_regex": regexes,
        "subset_for_metrics": language,
        "relaxed": False,
        "category": category,
        **get_mcq_fields(entry),
    }


def main(args):
    invalid = set(args.languages) - set(SUPPORTED_LANGUAGES)
    if invalid:
        raise ValueError(f"Unsupported languages: {invalid}")
    datasets = download_mmmlu_datasets(args.languages)

    data_dir = Path(__file__).absolute().parent
    output_file = data_dir / "test.jsonl"
    with open(output_file, "wt", encoding="utf-8") as fout:
        for language, examples in datasets.items():
            for entry in examples[: args.limit]:
                entry = format_entry(entry=entry, language=language)
                json.dump(entry, fout, ensure_ascii=False)
                fout.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--languages",
        default=SUPPORTED_LANGUAGES,
        nargs="+",
        help="Languages to process.",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Limit the number of examples to process.",
    )
    args = parser.parse_args()
    main(args)
