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
import os
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # for utils.py
from utils import assert_all, load_json, soft_assert  # noqa: E402

# Reasonable number of timeouts per file
MAX_TIMEOUTS_PER_FILE = 10

RANGE_CONSTRAINTS = {
    "aime25": {"pass@1[avg-of-16]": (93.33, 100.0), "majority@16": (100.0, 100.0), "pass@16": (100.0, 100.0)},
}


def parse_timeout_counts(eval_file: Path) -> int:
    """Return the total sandbox timeout count from the async jsonl output."""

    total_timeouts = 0
    with eval_file.open("rt", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            soft_assert("num_code_timeouts" in row, f"num_code_timeouts not found in {line}")
            total_timeouts += row.get("num_code_timeouts", 0)
    return total_timeouts


def check_timeouts(eval_dir: str):
    eval_dir = Path(eval_dir) / "eval-results" / "aime25"

    def extract_seed(p: Path) -> int:
        match = re.search(r"output-rs(\d+)\.jsonl", p.name)
        if not match:
            raise ValueError(f"Unexpected filename format: {p.name}")
        return int(match.group(1))

    timeout_files = sorted(eval_dir.glob("output-rs*.jsonl"), key=extract_seed)

    total_timeouts = 0
    for output_path in timeout_files:
        file_timeouts = parse_timeout_counts(output_path)
        total_timeouts += file_timeouts
        print(f"{output_path.name}: num_code_timeouts={file_timeouts}")

    if timeout_files:
        print(f"Total sandbox_code_timeouts across files: {total_timeouts}")
        allowed_timeouts = MAX_TIMEOUTS_PER_FILE * len(timeout_files)
        soft_assert(
            total_timeouts <= allowed_timeouts,
            (
                "Code execution timeouts regressed: "
                f"observed {total_timeouts}, allowed <= {allowed_timeouts} "
                f"({len(timeout_files)} files * {MAX_TIMEOUTS_PER_FILE} max each)"
            ),
        )


def check_results(eval_dir: str):
    f = os.path.join(eval_dir, "eval-results", "aime25", "metrics.json")
    eval_results = load_json(f)

    for benchmark, expected_metrics in RANGE_CONSTRAINTS.items():
        for metric, (lo, hi) in expected_metrics.items():
            accuracy = eval_results[benchmark][metric]["symbolic_correct"]
            soft_assert(lo <= accuracy <= hi, f"{benchmark}: {metric} {accuracy}% out of range [{lo}%, {hi}%]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, help="Workspace directory containing eval results")
    args = ap.parse_args()

    check_results(args.workspace)
    check_timeouts(args.workspace)

    assert_all()


if __name__ == "__main__":
    main()
