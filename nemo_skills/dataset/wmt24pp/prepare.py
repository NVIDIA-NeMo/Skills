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
from langcodes import Language


def write_data_to_file(output_file, datasets, tgt_languages):
    with open(output_file, "wt", encoding="utf-8") as fout:
        for tgt_lang in tgt_languages:
            for src, tgt in zip(datasets[tgt_lang]["source"], datasets[tgt_lang]["target"], strict=True):
                json_dict = {
                    "text": src,
                    "translation": tgt,
                    "source_language": "en",
                    "target_language": tgt_lang,
                    "source_lang_name": "English",
                    "target_lang_name": Language(tgt_lang[:2]).display_name(),
                }
                json.dump(json_dict, fout)
                fout.write("\n")


def main(args):
    datasets = {}
    for lang in args.target_languages:
        datasets[lang] = load_dataset("google/wmt24pp", f"en-{lang}")["train"]

    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / f"{args.split}.jsonl"
    write_data_to_file(output_file, datasets, tgt_languages=args.target_languages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("test",), help="Dataset split to process.")
    parser.add_argument(
        "--target_languages",
        default=["de_DE", "es_MX", "fr_FR", "it_IT", "ja_JP"],
        nargs="+",
        help="Languages to translate to.",
    )
    args = parser.parse_args()
    main(args)
