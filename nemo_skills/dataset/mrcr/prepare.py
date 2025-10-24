# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import tiktoken
from datasets import load_dataset
from tqdm import tqdm

"""
Usage
# default. setup is all.
python prepare.py

# prepare subset needle2_128k.
python prepare.py --max_context_window 131072 --needles_subset 2 --setup needle2_128k
python prepare.py --max_context_window 131072 --needles_subset 2 4 --setup needle2_needle_4_128k
"""


def count_n_tokens(messages: list[dict]) -> int:
    """
    Follow the official way to count tokens in messages.
    with tokenizer o200k_base
    """
    enc = tiktoken.get_encoding("o200k_base")
    return sum([len(enc.encode(m["content"])) for m in messages])


def write_data_to_file(output_file, data, max_context_window, needles_subset):
    with open(output_file, "wt", encoding="utf-8") as fout:
        for idx, entry in tqdm(enumerate(data), desc=f"Writing {output_file.name}"):
            messages = json.loads(entry.pop("prompt"))

            if entry["n_needles"] not in needles_subset:
                print(f"Skipping {idx} because it has {entry['n_needles']} needle")
                continue

            # find n_tokens
            n_tokens = count_n_tokens(messages)
            if max_context_window is not None:
                if n_tokens > max_context_window:
                    print(f"Skipping {idx} because it has {n_tokens} tokens")
                    continue

            entry["messages"] = messages
            entry["expected_answer"] = entry.pop("answer")
            entry["n_tokens"] = n_tokens
            json.dump(entry, fout)
            fout.write("\n")


def get_mrcr_data(needles_subset, setup, max_context_window):
    dataset = load_dataset("openai/mrcr")["train"]
    data_dir = Path(__file__).absolute().parent

    output_file = data_dir / f"{setup}.jsonl"
    write_data_to_file(output_file, dataset, max_context_window, needles_subset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MRCR dataset.")
    parser.add_argument(
        "--max_context_window",
        type=int,
        default=None,
        help="Maximum context window size.",
    )
    parser.add_argument(
        "--needles_subset",
        nargs="+",
        type=int,
        choices=[2, 4, 8],
        default=[2, 4, 8],
        help="Needles subset to include.",
    )

    parser.add_argument(
        "--setup",
        type=str,
        default="all",
        help="setup name. e.g. all or <needle2>_<128k>",
    )

    args = parser.parse_args()

    print(f"Preparing MRCR dataset with additional arguments: {args}")
    get_mrcr_data(args.needles_subset, args.setup, args.max_context_window)
    print(f"MRCR dataset preparation with setup {args.setup} completed. Use --split={args.setup} to evaluate!")
