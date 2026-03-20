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
import urllib.request
from pathlib import Path

import datasets

INSTRUCTIONS_URL = "https://raw.githubusercontent.com/QwenLM/PolyMath/main/instruction.py"


def _load_instructions(url: str) -> tuple[dict, dict, dict]:
    with urllib.request.urlopen(url) as response:
        ns = {}
        exec(compile(response.read().decode("utf-8"), url, "exec"), ns)
        return ns["language_dic"], ns["query_dic"], ns["language_control"]


language_dic, query_dic, language_control = _load_instructions(INSTRUCTIONS_URL)

difficulty_level_dic = {"low": 1, "medium": 2, "high": 4, "top": 8}

QUESTION_TEMPLATE = """
{question}

{instruction} {lang_control}
""".strip()


SUPPORTED_LANGUAGES = list(language_dic.keys())


def format_entry(entry: dict, language: str, difficulty: str, language_control_mode: str = None) -> dict:
    problem = entry["question"]
    instruction = query_dic[language]
    lang_control = language_control[language_control_mode][language] if language_control_mode else ""
    question = QUESTION_TEMPLATE.format(question=problem, instruction=instruction, lang_control=lang_control).strip()
    return {
        "question": question,
        "problem": problem,  # keep the original problem text for the judge
        "expected_answer": entry["answer"],
        "subset_for_metrics": language,
        "difficulty": difficulty,
        "weight": difficulty_level_dic[difficulty],
        "language_control_mode": language_control_mode,
    }


def main(args):
    invalid_langs = set(args.languages) - set(SUPPORTED_LANGUAGES)
    if invalid_langs:
        raise ValueError(f"Unsupported languages: {invalid_langs}. Supported: {SUPPORTED_LANGUAGES}")

    data_dir = Path(__file__).absolute().parent
    output_file = data_dir / "test.jsonl"

    all_entries = []
    for language in args.languages:
        for difficulty in difficulty_level_dic:
            print(f"Processing {language}/{difficulty}...")
            ds = datasets.load_dataset("Qwen/PolyMath", name=language, split=difficulty)
            for entry in ds:
                formatted = format_entry(
                    entry,
                    language=language,
                    difficulty=difficulty,
                    language_control_mode=args.language_control,
                )
                all_entries.append(formatted)

    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in all_entries:
            json.dump(entry, fout, ensure_ascii=False)
            fout.write("\n")

    print(f"Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare PolyMath multilingual math benchmark.")
    parser.add_argument(
        "--languages",
        default=SUPPORTED_LANGUAGES,
        nargs="+",
        help=f"Languages to include. Supported: {SUPPORTED_LANGUAGES}",
    )
    parser.add_argument(
        "--language_control",
        default=None,
        choices=list(language_control.keys()),
        help="Language control mode. One of: forcing_raw, forcing_en, forcing_prefer. Disabled by default.",
    )
    args = parser.parse_args()
    main(args)
