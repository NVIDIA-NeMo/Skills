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
import importlib.util
import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def _load_utils():
    utils_file = Path(__file__).absolute().parent / "aime24_x_utils.py"
    spec = importlib.util.spec_from_file_location("aime24_x_utils", utils_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_utils = _load_utils()
EN_INSTRUCTION = _utils.EN_INSTRUCTION
MATH_INSTRUCTIONS = _utils.MATH_INSTRUCTIONS
SUPPORTED_LANGUAGES = _utils.SUPPORTED_LANGUAGES


def format_entry(entry, lang, prompt_language):
    instruction = MATH_INSTRUCTIONS[lang] if prompt_language == "target" else EN_INSTRUCTION
    return {
        "question": f"{instruction}\n\n{entry['problem']}",
        "problem": entry["problem"],
        "expected_answer": entry["expected_answer"],
        "subset_for_metrics": lang,
        "target_language": lang,
    }


def main(args):
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / "test.jsonl"

    all_entries = []
    for lang in args.languages:
        print(f"Loading language: {lang}")
        ds = load_dataset("nvidia/Nemotron-Multilinugual-Eval-AIME24", split=lang)
        for entry in tqdm(ds, desc=f"Collecting {lang}"):
            all_entries.append(format_entry(entry, lang, args.prompt_language))

    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in all_entries:
            json.dump(entry, fout, ensure_ascii=False)
            fout.write("\n")

    print(f"Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default=SUPPORTED_LANGUAGES, nargs="+")
    parser.add_argument(
        "--prompt_language",
        default="target",
        choices=["target", "en"],
        help="Language for the prompt instruction. 'target' uses the data language; 'en' uses English.",
    )
    args = parser.parse_args()
    main(args)
