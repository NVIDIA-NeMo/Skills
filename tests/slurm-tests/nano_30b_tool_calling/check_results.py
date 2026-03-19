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

"""Check results for nano_30b_tool_calling SLURM test.

Validates:
  1. Tool calls were actually made (num_tool_calls > 0)
  2. Accuracy is within expected range
  3. Code execution timeouts are within acceptable limits
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # for utils.py
from utils import assert_all, load_json, soft_assert  # noqa: E402

MATH_BENCHMARKS = ["aime24", "aime25"]

# Baseline (2026-03-12, Nano-30B, stdio transport, max_tool_calls=100):
#   aime24: pass@1=95.00%, majority@16=100.00%, pass@16=100.00%, timeouts=12
#   aime25: pass@1=96.67%, majority@16=100.00%, pass@16=100.00%, timeouts=82
MATH_METRIC_RANGES = {
    "aime24": {
        "pass@1[avg-of-16]": (90.0, 100.0),
        "majority@16": (96.67, 100.0),
        "pass@16": (96.67, 100.0),
    },
    "aime25": {
        "pass@1[avg-of-16]": (93.33, 100.0),
        "majority@16": (96.67, 100.0),
        "pass@16": (96.67, 100.0),
    },
}

# At least this fraction of samples should have made tool calls
MIN_TOOL_CALL_FRACTION = 0.3

# Maximum timeouts allowed per benchmark
# Baseline (aime24=12, aime25=82)
MAX_TIMEOUTS = {
    "aime24": 30,
    "aime25": 100,
}

# Strings in tool response content that indicate a sandbox timeout
TIMEOUT_INDICATORS = [
    "execution timed out",
    "timed out",
    "process_status.*timeout",
    "TimeoutError",
]


def check_tool_usage(eval_dir: str):
    """Verify that tool calls were actually made during generation."""
    total_samples = 0
    samples_with_tools = 0

    for benchmark in MATH_BENCHMARKS:
        bench_dir = Path(eval_dir) / "eval-results" / benchmark
        output_files = sorted(bench_dir.glob("output-rs*.jsonl"))
        soft_assert(len(output_files) > 0, f"No output files found in {bench_dir}")

        for output_path in output_files:
            with output_path.open("rt", encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    total_samples += 1
                    if row.get("num_tool_calls", 0) > 0:
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


def check_timeouts(eval_dir: str):
    """Check that total code execution timeouts are within acceptable limits.

    Since the tool_calling path doesn't have a dedicated num_code_timeouts field
    (unlike CodeExecutionWrapper), we scan tool response messages in the conversation
    for timeout indicators from the sandbox.
    """
    timeout_pattern = re.compile("|".join(TIMEOUT_INDICATORS), re.IGNORECASE)

    for benchmark in MATH_BENCHMARKS:
        bench_dir = Path(eval_dir) / "eval-results" / benchmark
        output_files = sorted(bench_dir.glob("output-rs*.jsonl"))
        bench_timeouts = 0

        for output_path in output_files:
            file_timeouts = 0
            with output_path.open("rt", encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    for msg in row.get("conversation", []):
                        if msg.get("role") == "tool":
                            content = str(msg.get("content", ""))
                            if timeout_pattern.search(content):
                                file_timeouts += 1
            bench_timeouts += file_timeouts
            if file_timeouts > 0:
                print(f"{benchmark}/{output_path.name}: num_code_timeouts={file_timeouts}")

        allowed = MAX_TIMEOUTS[benchmark]
        print(f"{benchmark} total code_timeouts: {bench_timeouts} (allowed: {allowed})")
        soft_assert(
            bench_timeouts <= allowed,
            f"{benchmark}: code execution timeouts regressed: observed {bench_timeouts}, allowed <= {allowed}",
        )


def check_math_tool_calling(eval_dir: str):
    """Check accuracy metrics for math benchmarks with tool calling."""
    for benchmark in MATH_BENCHMARKS:
        f = os.path.join(eval_dir, "eval-results", benchmark, "metrics.json")
        data = load_json(f)

        for metric, (lo, hi) in MATH_METRIC_RANGES[benchmark].items():
            val = float(data[benchmark][metric]["symbolic_correct"])
            print(f"{benchmark}/{metric}: {val}%")
            soft_assert(lo <= val <= hi, f"{benchmark}: {metric} {val}% out of range [{lo}%, {hi}%]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, help="Workspace directory containing results")
    args = ap.parse_args()

    eval_root = Path(args.workspace)

    check_tool_usage(eval_root)
    check_timeouts(eval_root)
    check_math_tool_calling(eval_root)

    assert_all()


if __name__ == "__main__":
    main()
