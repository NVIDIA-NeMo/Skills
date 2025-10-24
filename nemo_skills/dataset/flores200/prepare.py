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


def write_data_to_file(output_file, datasets, src_languages, tgt_languages):
    with open(output_file, "wt", encoding="utf-8") as fout:
        for src_lang in src_languages:
            for tgt_lang in tgt_languages:
                if src_lang != tgt_lang:
                    for src, tgt in zip(datasets[src_lang], datasets[tgt_lang], strict=True):
                        json_dict = {
                            "text": src,
                            "translation": tgt,
                            "source_language": src_lang,
                            "target_language": tgt_lang,
                            "source_lang_name": Language(src_lang).display_name(),
                            "target_lang_name": Language(tgt_lang).display_name(),
                        }
                        json.dump(json_dict, fout)
                        fout.write("\n")


def main(args):
    all_languages = list(set(args.source_languages).union(set(args.target_languages)))

    datasets = {}
    for lang in all_languages:
        iso_639_3 = Language(lang).to_alpha3()
        iso_15924 = Language(lang).maximize().script
        lang_code = f"{iso_639_3}_{iso_15924}"
        datasets[lang] = load_dataset("openlanguagedata/flores_plus", lang_code, split=args.split)["text"]

    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / f"{args.split}.jsonl"
    write_data_to_file(output_file, datasets, src_languages=args.source_languages, tgt_languages=args.target_languages)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="devtest", choices=("dev", "devtest"), help="Dataset split to process.")
    parser.add_argument(
        "--source_languages",
        default=["en", "de", "es", "fr", "it", "ja"],
        nargs="+",
        help="Languages to translate from.",
    )
    parser.add_argument(
        "--target_languages",
        default=["en", "de", "es", "fr", "it", "ja"],
        nargs="+",
        help="Languages to translate to.",
    )
    args = parser.parse_args()
    main(args)
