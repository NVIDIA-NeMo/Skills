# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # for utils.py
from utils import assert_all, load_json, soft_assert  # noqa: E402

METRIC_RANGES = {
    "vllm": {"aime25": {"pass@1": (88.0, 92.0)}},
    "sglang": {"aime25": {"pass@1": (88.0, 92.0)}},
    "trtllm": {"aime25": {"pass@1": (88.0, 92.0)}},
}
TIR_METRIC_RANGES = {
    "vllm": {"aime25": {"pass@1": (92.0, 100.0)}},
    "sglang": {"aime25": {"pass@1": (92.0, 100.0)}},
    "trtllm": {"aime25": {"pass@1": (92.0, 100.0)}},
}
MIN_TOOL_CALL_FRACTION = 0.05


def check_results(workspace: str, backend: str):
    metrics_path = os.path.join(workspace, backend, "eval-results", "aime25", "metrics.json")
    metrics = load_json(metrics_path)

    for benchmark, expected_metrics in METRIC_RANGES[backend].items():
        for metric, (lo, hi) in expected_metrics.items():
            accuracy = float(metrics[benchmark][metric]["symbolic_correct"])
            soft_assert(
                lo <= accuracy <= hi,
                f"{backend}/{benchmark}: {metric} {accuracy}% out of range [{lo}%, {hi}%]",
            )


def iter_output_rows(bench_dir: Path):
    output_files = sorted(bench_dir.glob("output-rs*.jsonl"))
    soft_assert(len(output_files) > 0, f"No output files found in {bench_dir}")
    for output_path in output_files:
        with output_path.open("rt", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                yield json.loads(line)


def check_tir_results(workspace: str, backend: str):
    metrics_path = os.path.join(workspace, f"{backend}_tir", "eval-results", "aime25", "metrics.json")
    metrics = load_json(metrics_path)

    for benchmark, expected_metrics in TIR_METRIC_RANGES[backend].items():
        for metric, (lo, hi) in expected_metrics.items():
            accuracy = float(metrics[benchmark][metric]["symbolic_correct"])
            soft_assert(
                lo <= accuracy <= hi,
                f"{backend}_tir/{benchmark}: {metric} {accuracy}% out of range [{lo}%, {hi}%]",
            )

    bench_dir = Path(workspace) / f"{backend}_tir" / "eval-results" / "aime25"
    total_samples = 0
    samples_with_tools = 0
    samples_with_tool_messages = 0
    for row in iter_output_rows(bench_dir):
        total_samples += 1
        soft_assert("num_tool_calls" in row, f"Missing num_tool_calls in {backend}_tir output row")
        soft_assert("conversation" in row, f"Missing conversation in {backend}_tir output row")
        if "num_tool_calls" not in row or "conversation" not in row:
            continue
        if row["num_tool_calls"] > 0:
            samples_with_tools += 1
        has_tool_message = False
        for msg in row["conversation"]:
            if isinstance(msg, dict) and msg.get("role") == "tool":
                has_tool_message = True
                break
        if has_tool_message:
            samples_with_tool_messages += 1

    soft_assert(total_samples > 0, f"No samples found in {backend}_tir outputs")
    if total_samples > 0:
        tool_fraction = samples_with_tools / total_samples
        soft_assert(
            tool_fraction >= MIN_TOOL_CALL_FRACTION,
            f"{backend}_tir: too few samples used tools: {tool_fraction:.1%} < {MIN_TOOL_CALL_FRACTION:.0%}",
        )
        soft_assert(samples_with_tool_messages > 0, f"{backend}_tir: no samples contained tool messages")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, help="Workspace directory containing eval results")
    args = ap.parse_args()

    for backend in METRIC_RANGES:
        check_results(args.workspace, backend)
        check_tir_results(args.workspace, backend)
    assert_all()


if __name__ == "__main__":
    main()
