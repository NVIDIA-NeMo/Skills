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
import os
from pathlib import Path

from datasets import load_dataset

_CONTEXT_FILES_BLOCK_TEMPLATE = """
--- file: {path}
```{fence}
{content}
```
"""


def _fence_for_path(path: str) -> str:
    p = path.lower()
    if p.endswith((".cu", ".cuh")):
        return "cuda"
    if p.endswith((".cc", ".cpp", ".cxx")):
        return "cpp"
    if p.endswith(".c"):
        return "c"
    if p.endswith(".h") or p.endswith(".hpp"):
        return "h"
    # Default to plaintext if unknown
    return ""


def _format_context_files_block(context_files: list[dict[str, str]]) -> str:
    blocks: list[str] = []
    for source in context_files:
        if "path" not in source or "content" not in source:
            continue

        fence = _fence_for_path(source["path"])
        blocks.append(
            _CONTEXT_FILES_BLOCK_TEMPLATE.format(path=source["path"], fence=fence, content=source["content"])
        )
    return "".join(blocks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare nvidia/compute-eval dataset")
    parser.add_argument(
        "--release",
        type=str,
        default=None,
        help="Release to download (e.g., '2025-1', '2025-2'). If not specified, downloads default release.",
    )

    args = parser.parse_args()

    token = os.getenv("HF_TOKEN", None)
    if not token:
        print("Error: HF_TOKEN environment variable not set. Please set it to access the dataset.")
        exit(1)

    dataset = load_dataset("nvidia/compute-eval", args.release, token=token)
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)

    with open(data_dir / "eval.jsonl", "wt", encoding="utf-8") as f:
        for item in dataset["eval"]:
            record = {
                "problem": item,
                "task_id": item["task_id"],
                "problem_prompt": item["prompt"],
                "build_command": item["build_command"],
                "context_files_block": _format_context_files_block(item["context_files"]),
            }

            # Dumping using default=str to handle datetime serialization from the problem records
            f.write(json.dumps(record, default=str) + "\n")
