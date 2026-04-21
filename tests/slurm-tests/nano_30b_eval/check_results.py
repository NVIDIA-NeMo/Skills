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

"""Check results for nano_30b_eval SLURM benchmark suite."""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # for utils.py
from utils import assert_all, load_json, soft_assert  # noqa: E402

NO_TOOLS_METRICS = {
    "aime25": ("pass@1[avg-of-4]", "symbolic_correct", (84.0, 94.0)),
    "gpqa": ("pass@1[avg-of-4]", "symbolic_correct", (69.0, 76.0)),
    "mmlu-pro": ("pass@1", "symbolic_correct", (74.0, 82.0)),
    "livecodebench": ("pass@1[avg-of-4]", "accuracy", (62.0, 72.0)),
    "scicode": ("pass@1[avg-of-4]", "subtask_accuracy", (28.0, 38.0)),
    "hle": ("pass@1", "judge_correct", (8.0, 14.0)),
}

WITH_TOOLS_METRICS = {
    "aime25": ("pass@1[avg-of-4]", "symbolic_correct", (88.0, 100.0)),
    "gpqa": ("pass@1[avg-of-4]", "symbolic_correct", (72.0, 78.0)),
    "hle": ("pass@1", "judge_correct", (13.0, 19.0)),
}

FORMAL_MATH_METRICS = {
    "minif2f_pass1": ("minif2f", "pass@1[avg-of-32]", "lean4_correct", (42.0, 58.0)),
    "minif2f_pass32": ("minif2f", "pass@32", "lean4_correct", (72.0, 88.0)),
}

TOOL_BENCHMARKS = ["aime25", "gpqa", "hle"]
MIN_TOOL_CALL_FRACTION = 0.05
MAX_TIMEOUTS = {
    "aime25": 200,
    "gpqa": 1000,
    "hle": 4000,
}
TIMEOUT_INDICATORS = [
    "execution timed out",
    "timed out",
    "process_status.*timeout",
    "TimeoutError",
]


def load_metrics_block(metrics_path: Path, benchmark: str):
    data = load_json(metrics_path)
    soft_assert(benchmark in data, f"Missing benchmark {benchmark} in {metrics_path}")
    return data[benchmark]


def normalize_percent(value: float) -> float:
    return value * 100.0 if 0.0 <= value <= 1.0 else value


def check_metric_group(eval_dir: Path, metric_config: dict[str, tuple[str, str, tuple[float, float]]]):
    for benchmark, (agg_key, field, (lo, hi)) in metric_config.items():
        metrics_path = eval_dir / "eval-results" / benchmark / "metrics.json"
        metrics = load_metrics_block(metrics_path, benchmark)
        soft_assert(agg_key in metrics, f"Missing aggregation key {agg_key} in {metrics_path}")
        soft_assert(field in metrics[agg_key], f"Missing field {field} in {metrics_path}")
        value = normalize_percent(float(metrics[agg_key][field]))
        print(f"{eval_dir.name}/{benchmark}/{agg_key}/{field}: {value}")
        soft_assert(lo <= value <= hi, f"{benchmark}: {field}={value} out of range [{lo}, {hi}]")


def iter_output_rows(bench_dir: Path):
    output_files = sorted(bench_dir.glob("output-rs*.jsonl"))
    soft_assert(len(output_files) > 0, f"No output files found in {bench_dir}")

    for output_path in output_files:
        with output_path.open("rt", encoding="utf-8") as fin:
            for line in fin:
                if not line.strip():
                    continue
                yield output_path, json.loads(line)


def check_tool_usage(eval_dir: Path):
    total_samples = 0
    samples_with_tools = 0
    samples_with_tool_messages = 0

    for benchmark in TOOL_BENCHMARKS:
        bench_dir = eval_dir / "eval-results" / benchmark
        for _, row in iter_output_rows(bench_dir):
            total_samples += 1
            if row.get("num_tool_calls", 0) > 0:
                samples_with_tools += 1
            if any(msg.get("role") == "tool" for msg in row.get("conversation", [])):
                samples_with_tool_messages += 1

    soft_assert(total_samples > 0, "No samples found in with_tools outputs")
    tool_fraction = samples_with_tools / total_samples
    print(
        f"with_tools/tool_usage: {samples_with_tools}/{total_samples} "
        f"samples used tools ({tool_fraction:.1%}), {samples_with_tool_messages} had tool messages"
    )
    soft_assert(
        tool_fraction >= MIN_TOOL_CALL_FRACTION,
        f"Too few samples used tools: {tool_fraction:.1%} < {MIN_TOOL_CALL_FRACTION:.0%}",
    )
    soft_assert(samples_with_tool_messages > 0, "No samples contained tool messages")


def check_timeouts(eval_dir: Path):
    timeout_pattern = re.compile("|".join(TIMEOUT_INDICATORS), re.IGNORECASE)

    for benchmark in TOOL_BENCHMARKS:
        bench_dir = eval_dir / "eval-results" / benchmark
        bench_timeouts = 0

        for output_path in sorted(bench_dir.glob("output-rs*.jsonl")):
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


def check_formal_math(eval_dir: Path):
    for label, (benchmark, agg_key, field, (lo, hi)) in FORMAL_MATH_METRICS.items():
        metrics_path = eval_dir / "eval-results" / benchmark / "metrics.json"
        metrics = load_metrics_block(metrics_path, benchmark)
        soft_assert(agg_key in metrics, f"Missing aggregation key {agg_key} in {metrics_path}")
        soft_assert(field in metrics[agg_key], f"Missing field {field} in {metrics_path}")
        value = normalize_percent(float(metrics[agg_key][field]))
        print(f"formal_math/{label}/{agg_key}/{field}: {value}")
        soft_assert(lo <= value <= hi, f"{label}: {field}={value} out of range [{lo}, {hi}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, help="Workspace directory containing results")
    args = ap.parse_args()

    eval_root = Path(args.workspace)

    print("=== no_tools ===")
    check_metric_group(eval_root / "no_tools", NO_TOOLS_METRICS)

    print("\n=== with_tools ===")
    check_metric_group(eval_root / "with_tools", WITH_TOOLS_METRICS)
    check_tool_usage(eval_root / "with_tools")
    check_timeouts(eval_root / "with_tools")

    print("\n=== formal_math ===")
    check_formal_math(eval_root / "formal_math")

    assert_all()


if __name__ == "__main__":
    main()
