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
from pathlib import Path

# Hard-coded accuracy ranges for baseline and after-training results
# TODO: should we train for longer / generate more data? Variance is really high
RANGE_CONSTRAINTS = {
    "after_training": {
        "aime24": {"pass@1[avg-of-8]": (17.0, 30.0), "majority@8": (28.33, 48.33)},
        "aime25": {"pass@1[avg-of-8]": (15.0, 27.5), "majority@8": (21.22, 42.22)},
    },
    "baseline": {
        "aime24": {"pass@1[avg-of-8]": (6.25, 18.25), "majority@8": (13.33, 24.33)},
        "aime25": {"pass@1[avg-of-8]": (8.75, 18.75), "majority@8": (11.67, 24.33)},
    },
}


def load_json(path: Path):
    """Load a JSON file from the given path."""
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r") as f:
        return json.load(f)


def get_aime_symbolic(d: dict, bench_key: str, metric_key: str) -> float:
    """Extract the value of a specific metric and convert to float."""
    return float(d[bench_key][metric_key]["symbolic_correct"])


def in_range(value: float, lo: float, hi: float) -> bool:
    """Return True if value is within [lo, hi] inclusive."""
    return lo <= value <= hi


def check_benchmark(benchmark: str, baseline_results: dict, after_training_results: dict):
    """
    Validate one benchmark:
      - baseline accuracy must be within its allowed range
      - after-training accuracy must be within its allowed range
    """
    for metric in ["pass@1[avg-of-8]", "majority@8"]:
        baseline_acc = get_aime_symbolic(baseline_results, benchmark, metric)
        after_acc = get_aime_symbolic(after_training_results, benchmark, metric)

        lo_b, hi_b = RANGE_CONSTRAINTS["baseline"][benchmark][metric]
        lo_a, hi_a = RANGE_CONSTRAINTS["after_training"][benchmark][metric]

        assert in_range(baseline_acc, lo_b, hi_b), (
            f"{benchmark}: baseline {baseline_acc}% out of range [{lo_b}%, {hi_b}%] for metric {metric}"
        )
        assert in_range(after_acc, lo_a, hi_a), (
            f"{benchmark}: after_training {after_acc}% out of range [{lo_a}%, {hi_a}%] for metric {metric}"
        )


def main():
    ap = argparse.ArgumentParser(
        description="Compare after-training vs baseline on AIME24/25 (metric: pass@1[avg-of-8].symbolic_correct)."
    )
    ap.add_argument("--workspace", required=True, help="Workspace directory containing eval results.")
    args = ap.parse_args()

    for benchmark in ("aime24", "aime25"):
        common_path = Path(args.workspace) / "evals"
        baseline_results = load_json(common_path / "baseline" / "eval-results" / benchmark / "metrics.json")
        after_training_results = load_json(
            common_path / "after-training" / "eval-results" / benchmark / "metrics.json"
        )
        check_benchmark(benchmark, baseline_results, after_training_results)

    print("All checks passed.")


if __name__ == "__main__":
    main()
