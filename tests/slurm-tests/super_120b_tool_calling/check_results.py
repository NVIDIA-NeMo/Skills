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

"""Check results for super_120b_tool_calling SLURM test.

Validates:
  1. Tool calls were actually made (num_tool_calls > 0)
  2. Accuracy is within expected range for AIME24
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # for utils.py
from utils import assert_all, load_json, soft_assert  # noqa: E402

# Accuracy ranges for AIME24 with tool calling (16 seeds)
# Baseline (2026-03-11, stdio transport): pass@1=88.33%, majority@16=95.00%, pass@16=100.00%
RANGE_CONSTRAINTS = {
    "aime24": {
        "pass@1[avg-of-16]": (80.0, 100.0),
        "majority@16": (86.67, 100.0),
        "pass@16": (93.33, 100.0),
    },
}

# At least this fraction of samples should have made tool calls
MIN_TOOL_CALL_FRACTION = 0.3


def check_tool_usage(workspace: str):
    """Verify that tool calls were actually made during generation."""
    gen_dir = Path(workspace) / "eval-results" / "aime24"

    output_files = sorted(gen_dir.glob("output-rs*.jsonl"))
    soft_assert(len(output_files) > 0, f"No output files found in {gen_dir}")

    total_samples = 0
    samples_with_tools = 0

    for output_path in output_files:
        with output_path.open("rt", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                row = json.loads(line)
                total_samples += 1
                num_tool_calls = row.get("num_tool_calls", 0)
                if num_tool_calls > 0:
                    samples_with_tools += 1

    if total_samples > 0:
        fraction = samples_with_tools / total_samples
        print(f"Tool usage: {samples_with_tools}/{total_samples} samples ({fraction:.1%})")
        soft_assert(
            fraction >= MIN_TOOL_CALL_FRACTION,
            f"Too few samples used tools: {fraction:.1%} < {MIN_TOOL_CALL_FRACTION:.0%} "
            f"({samples_with_tools}/{total_samples})",
        )
    else:
        soft_assert(False, "No samples found in output files")


def check_accuracy(workspace: str):
    """Check accuracy metrics are within expected range."""
    metrics_path = os.path.join(workspace, "eval-results", "aime24", "metrics.json")
    eval_results = load_json(metrics_path)

    for benchmark, expected_metrics in RANGE_CONSTRAINTS.items():
        for metric, (lo, hi) in expected_metrics.items():
            accuracy = eval_results[benchmark][metric]["symbolic_correct"]
            print(f"{benchmark}/{metric}: {accuracy}%")
            soft_assert(
                lo <= accuracy <= hi,
                f"{benchmark}: {metric} {accuracy}% out of range [{lo}%, {hi}%]",
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, help="Workspace directory containing results")
    args = ap.parse_args()

    check_tool_usage(args.workspace)
    check_accuracy(args.workspace)

    assert_all()


if __name__ == "__main__":
    main()
