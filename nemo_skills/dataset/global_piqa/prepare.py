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

from nemo_skills.dataset.global_piqa.global_piqa_utils import (
    EXTRACT_REGEX,
    Schema,
    digit_to_letter,
    get_mcq_fields,
    load_global_piqa_datasets,
    supported_languages,
)


def format_entry(entry: dict, language: str) -> dict:
    return {
        "expected_answer": digit_to_letter(entry[Schema.LABEL]),
        "extract_from_boxed": False,
        "extract_regex": EXTRACT_REGEX,
        "subset_for_metrics": language,
        "relaxed": False,
        **get_mcq_fields(entry),
    }


def main(args):
    invalid = set(args.languages) - set(supported_languages())
    if invalid:
        raise ValueError(f"Unsupported languages: {invalid}")
    datasets = load_global_piqa_datasets(args.languages)

    data_dir = Path(__file__).absolute().parent
    output_file = data_dir / "test.jsonl"
    with open(output_file, "wt", encoding="utf-8") as fout:
        for language in datasets:
            for entry in datasets[language]:
                entry = format_entry(entry=entry, language=language)
                json.dump(entry, fout, ensure_ascii=False)
                fout.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--languages",
        default=supported_languages(),
        nargs="+",
        help="Languages to process.",
    )
    args = parser.parse_args()
    main(args)
