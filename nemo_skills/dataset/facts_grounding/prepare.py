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

from datasets import load_dataset
from tqdm import tqdm


def write_data_to_file(output_file, examples):
    with open(output_file, "wt", encoding="utf-8") as fout:
        for row in tqdm(examples, desc=f"Writing {output_file.name}"):
            json.dump(row, fout, ensure_ascii=False)
            fout.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="test",
        choices=("test",),
        help="Dataset split to process.",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)

    # Download the FACTS Grounding examples
    ds = load_dataset("google/FACTS-grounding-public", "examples", split="public")

    examples = []
    for idx, sample in enumerate(ds):
        examples.append(
            {
                "id": f"facts_grounding_{idx}",
                "full_prompt": sample["full_prompt"],
                "user_request": sample["user_request"],
                "context_document": sample["context_document"],
            }
        )

    output_file = data_dir / f"{args.split}.jsonl"
    write_data_to_file(output_file, examples)

    # Download evaluation prompts used by the judge
    eval_ds = load_dataset("google/FACTS-grounding-public", "evaluation_prompts", split="prompts")
    eval_prompts = {}
    for item in eval_ds:
        eval_prompts[item["evaluation_method"]] = item["evaluation_prompt"]

    eval_prompts_file = data_dir / "eval_prompts.json"
    with open(eval_prompts_file, "wt", encoding="utf-8") as f:
        json.dump(eval_prompts, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(examples)} examples to {output_file}")
    print(f"Wrote {len(eval_prompts)} evaluation prompts to {eval_prompts_file}")
