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
    utils_file = Path(__file__).absolute().parent / "livecodebench_x_utils.py"
    spec = importlib.util.spec_from_file_location("livecodebench_x_utils", utils_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_utils = _load_utils()
CODEGEN_INSTRUCTIONS = _utils.CODEGEN_INSTRUCTIONS
EN_INSTRUCTION = _utils.EN_INSTRUCTION
SUPPORTED_LANGUAGES = _utils.SUPPORTED_LANGUAGES
SUPPORTED_VERSIONS = _utils.SUPPORTED_VERSIONS
VERSION_TO_FILENAME = _utils.VERSION_TO_FILENAME


def format_entry(entry, lang, prompt_language):
    instruction = CODEGEN_INSTRUCTIONS[lang] if prompt_language == "target" else EN_INSTRUCTION
    return {
        "question": f"{instruction}\n\n{entry['question']}",
        "problem": entry["question"],
        "task_id": entry["task_id"],
        "release_version": entry["release_version"],
        "subset_for_metrics": lang,
        "target_language": lang,
    }


def main(args):
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)

    for version in args.versions:
        output_file = data_dir / VERSION_TO_FILENAME[version]
        print(f"Processing version {version} → {output_file.name}")
        all_entries = []
        for lang in args.languages:
            print(f"  Loading language: {lang}")
            ds = load_dataset("nvidia/Nemotron-Multilinugual-Eval-LCB", version, split=lang)
            for entry in tqdm(ds, desc=f"Collecting {version}/{lang}"):
                all_entries.append(format_entry(entry, lang, args.prompt_language))
        with open(output_file, "wt", encoding="utf-8") as fout:
            for entry in all_entries:
                json.dump(entry, fout, ensure_ascii=False)
                fout.write("\n")
        print(f"Saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", default=SUPPORTED_LANGUAGES, nargs="+")
    parser.add_argument("--versions", default=SUPPORTED_VERSIONS, nargs="+")
    parser.add_argument(
        "--prompt_language",
        default="target",
        choices=["target", "en"],
        help="Language for the prompt instruction. 'target' uses the data language; 'en' uses English.",
    )
    args = parser.parse_args()
    main(args)
