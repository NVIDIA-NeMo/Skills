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
import re
from pathlib import Path
from typing import Callable

import tiktoken
from datasets import load_dataset
from tqdm import tqdm

"""
Usage
# default setup is "all" (all problem types, no context window filter)
python prepare.py

# prepare only "parents" problem type
python prepare.py --problem_types parents --setup parents

# prepare with a 128k context window limit
python prepare.py --max_context_window 131072 --setup 128k

# prepare BFS problems within 128k context
python prepare.py --max_context_window 131072 --problem_types bfs --setup bfs_128k

# use a HuggingFace tokenizer instead of tiktoken
python prepare.py --tokenizer meta-llama/Llama-3.1-8B-Instruct --max_context_window 131072 --setup 128k
"""


def build_tokenizer(tokenizer_name: str) -> Callable[[str], int]:
    """Return a callable that counts tokens in a string.

    Tries tiktoken first; if the name is not a valid tiktoken encoding,
    falls back to a HuggingFace AutoTokenizer.
    """
    try:
        enc = tiktoken.get_encoding(tokenizer_name)
        return lambda text: len(enc.encode(text))
    except ValueError:
        from transformers import AutoTokenizer

        hf_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        return lambda text: len(hf_tokenizer.encode(text, add_special_tokens=False))


def write_data_to_file(output_file, data, max_context_window, problem_types, count_tokens: Callable[[str], int]):
    with open(output_file, "wt", encoding="utf-8") as fout:
        for idx, entry in tqdm(enumerate(data), desc=f"Writing {output_file.name}"):
            if problem_types is not None and entry["problem_type"] not in problem_types:
                print(f"Skipping {idx} because problem_type={entry['problem_type']} not in {problem_types}")
                continue

            prompt_text = entry["prompt"]
            answer_nodes = entry["answer_nodes"]

            pattern = r"Perform a BFS from node (\S+) with depth (\d+)"
            replacement = r"Perform a BFS from node \1 and return only the nodes at exactly depth \2 (not nodes at intermediate depths)"
            prompt_text = re.sub(pattern, replacement, prompt_text)

            m = re.search(r"Find the parents of node ([^\s.]+)\.", prompt_text)
            node_id = m.group(1) if m else None
            if node_id is not None and node_id in answer_nodes:
                answer_nodes.remove(node_id)
                print(f"Removing {idx} sample with node {node_id} from answer_nodes because it is in the prompt")

            n_tokens = count_tokens(prompt_text)

            if max_context_window is not None and n_tokens > max_context_window:
                print(f"Skipping {idx} because it has {n_tokens} tokens (limit={max_context_window})")
                continue

            messages = [{"role": "user", "content": prompt_text}]

            output_entry = {
                "messages": messages,
                "expected_answer": json.dumps(sorted(answer_nodes)),
                "n_tokens": n_tokens,
                "prompt_chars": entry["prompt_chars"],
                "problem_type": entry["problem_type"],
            }
            json.dump(output_entry, fout)
            fout.write("\n")


def get_graphwalks_data(problem_types, setup, max_context_window, tokenizer_name):
    dataset = load_dataset("openai/graphwalks")["train"]
    data_dir = Path(__file__).absolute().parent

    count_tokens = build_tokenizer(tokenizer_name)
    output_file = data_dir / f"{setup}.jsonl"
    write_data_to_file(output_file, dataset, max_context_window, problem_types, count_tokens)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare GraphWalks dataset.")
    parser.add_argument(
        "--max_context_window",
        type=int,
        default=None,
        help="Maximum context window size in tokens. Samples exceeding this will be skipped.",
    )
    parser.add_argument(
        "--problem_types",
        nargs="+",
        type=str,
        default=None,
        help="Problem types to include (e.g. parents bfs). Defaults to all types.",
    )
    parser.add_argument(
        "--setup",
        type=str,
        default="all",
        help="Setup name used as the output filename, e.g. 'all', 'parents', 'bfs_128k'.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="cl100k_base",
        help=(
            "Tokenizer to use for counting tokens. Pass a tiktoken encoding name "
            "(e.g. 'cl100k_base', 'o200k_base') or a HuggingFace model id / local path "
            "(e.g. 'meta-llama/Llama-3.1-8B-Instruct')."
        ),
    )

    args = parser.parse_args()

    print(f"Preparing GraphWalks dataset with arguments: {args}")
    get_graphwalks_data(args.problem_types, args.setup, args.max_context_window, args.tokenizer_name)
    print(f"GraphWalks dataset preparation with setup '{args.setup}' completed. Use --split={args.setup} to evaluate!")
