# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

_tokenizer_cache = {}
postfix = "\n\nThe last line of your response should be in the following format: 'Answer: \\boxed{A/B/C/D}' (e.g. 'Answer: \\boxed{A}')."


def count_n_tokens(prompt: str, tokenizer_name: str) -> int:
    """Count tokens using tiktoken (e.g. cl100k_base) or HuggingFace AutoTokenizer (e.g. meta-llama/Llama-3.1-8B)."""
    if tokenizer_name not in _tokenizer_cache:
        try:
            _tokenizer_cache[tokenizer_name] = tiktoken.get_encoding(tokenizer_name)
        except ValueError:
            from transformers import AutoTokenizer

            _tokenizer_cache[tokenizer_name] = AutoTokenizer.from_pretrained(tokenizer_name)
    enc = _tokenizer_cache[tokenizer_name]
    if isinstance(enc, tiktoken.Encoding):
        return len(enc.encode(prompt))
    return len(enc.encode(prompt, add_special_tokens=False))


def write_data_to_file(output_file, data, tokenizer_name):
    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in tqdm(data, desc=f"Writing {output_file.name}"):
            messages = [{"role": "user", "content": entry["prompt"].strip() + postfix}]
            out = {
                "messages": messages,
                "expected_answer": entry["correct_letter"],
                "repo": entry["repo"],
                "prompt_goal": entry["prompt_goal"],
                "is_hard": entry["is_hard"],
                f"n_tokens_{tokenizer_name}": count_n_tokens(entry["prompt"].strip() + postfix, tokenizer_name),
            }
            json.dump(out, fout)
            fout.write("\n")


def prepare_longcodebench_data(setup, tokenizer_name):
    ds = load_dataset("json", data_files="hf://datasets/Steefano/LCB/LongCodeQA.zip")
    data = ds["train"]

    data_dir = Path(__file__).absolute().parent
    output_file = data_dir / f"{setup}.jsonl"
    write_data_to_file(output_file, data, tokenizer_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LongCodeBench LongCodeQA dataset.")
    parser.add_argument(
        "--setup",
        type=str,
        default="test",
        help="Setup name for the output file (default: test).",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="cl100k_base",
        help="Tokenizer name for counting tokens (tiktoken or HuggingFace model name).",
    )

    args = parser.parse_args()

    print(f"Preparing LongCodeBench LongCodeQA dataset with setup: {args.setup}")
    prepare_longcodebench_data(args.setup, args.tokenizer_name)
    print(f"LongCodeBench dataset preparation completed. Use --split={args.setup} to evaluate!")
