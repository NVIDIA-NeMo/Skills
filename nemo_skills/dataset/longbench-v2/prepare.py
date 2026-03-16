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

"""
Prepare the LongBench-v2 dataset for evaluation.

LongBench v2 is a multiple-choice benchmark (4 choices: A/B/C/D) covering
6 domains over long contexts (8k–2M words):
  - Single-document QA
  - Multi-document QA
  - Long in-context learning
  - Long-dialogue history understanding
  - Code repository understanding
  - Long structured data understanding

Dataset: https://huggingface.co/datasets/THUDM/LongBench-v2
Paper:   https://arxiv.org/abs/2412.15204

Usage
-----
# Prepare all 503 questions (default setup name: "test")
python prepare.py

# Prepare only hard questions
python prepare.py --difficulty hard --setup test_hard

# Prepare short-context hard questions
python prepare.py --difficulty hard --length short --setup test_hard_short

or via ns prepare_data:
    ns prepare_data --data_dir=/workspace/ns-data --cluster=local longbenchv2
"""

_tokenizer_cache = {}


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


def write_data_to_file(output_file: Path, data, difficulty, length, tokenizer_name):
    skipped = 0
    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in tqdm(data, desc=f"Writing {output_file.name}"):
            if difficulty is not None and entry["difficulty"] != difficulty:
                skipped += 1
                continue
            if length is not None and entry["length"] != length:
                skipped += 1
                continue

            # Store individual fields so the prompt config template can reference them
            # as {context}, {question}, {choice_A}, {choice_B}, {choice_C}, {choice_D}
            # (corresponding to $DOC$, $Q$, $C_A$, $C_B$, $C_C$, $C_D$ in the official template).
            record = {
                "index": entry["_id"],
                "context": entry["context"],
                "question": entry["question"],
                "choice_A": entry["choice_A"],
                "choice_B": entry["choice_B"],
                "choice_C": entry["choice_C"],
                "choice_D": entry["choice_D"],
                "expected_answer": entry["answer"],
                "domain": entry["domain"],
                "sub_domain": entry["sub_domain"],
                "difficulty": entry["difficulty"],
                "length": entry["length"],
                "context_tokens": count_n_tokens(entry["context"], tokenizer_name),
            }
            fout.write(json.dumps(record) + "\n")

    if skipped:
        print(f"Skipped {skipped} entries.")


def prepare_longbenchv2_data(setup: str, difficulty, length, tokenizer_name):
    # HuggingFace dataset uses a single "train" split that contains all 503 evaluation questions
    dataset = load_dataset("THUDM/LongBench-v2", split="train")

    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)

    output_file = data_dir / f"{setup}.jsonl"
    write_data_to_file(output_file, dataset, difficulty, length, tokenizer_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare LongBench-v2 dataset.")
    parser.add_argument(
        "--setup",
        type=str,
        default="test",
        help="Output file name (without .jsonl). Use --split=<setup> when evaluating.",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        choices=["easy", "hard"],
        default=None,
        help="Keep only questions of this difficulty level.",
    )
    parser.add_argument(
        "--length",
        type=str,
        choices=["short", "medium", "long"],
        default=None,
        help="Keep only questions in this context-length category.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="cl100k_base",
        help="tokenizer name",
    )
    args = parser.parse_args()

    print(f"Preparing LongBench-v2 dataset with args: {args}")
    prepare_longbenchv2_data(args.setup, args.difficulty, args.length, args.tokenizer_name)
    print(f"LongBench-v2 preparation complete. Use --split={args.setup} to evaluate!")
