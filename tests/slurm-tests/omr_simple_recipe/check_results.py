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
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # for utils.py
from utils import assert_all, load_json, soft_assert  # noqa: E402

# TODO: should we train for longer / generate more data? Variance is really high
RANGE_CONSTRAINTS = {
    "after_training": {
        "aime24": {"pass@1[avg-of-8]": (16.0, 30.0), "majority@8": (25.33, 48.33)},
        "aime25": {"pass@1[avg-of-8]": (15.0, 27.5), "majority@8": (20.22, 42.22)},
    },
    "baseline": {
        "aime24": {"pass@1[avg-of-8]": (6.25, 18.25), "majority@8": (13.33, 25.33)},
        "aime25": {"pass@1[avg-of-8]": (8.75, 18.75), "majority@8": (11.67, 24.33)},
    },
}


def check_results(benchmark: str, baseline_results: dict, after_training_results: dict, backend: str):
    for metric in ["pass@1[avg-of-8]", "majority@8"]:
        baseline_acc = baseline_results[benchmark][metric]["symbolic_correct"]
        after_acc = after_training_results[benchmark][metric]["symbolic_correct"]

        lo_b, hi_b = RANGE_CONSTRAINTS["baseline"][benchmark][metric]
        lo_a, hi_a = RANGE_CONSTRAINTS["after_training"][benchmark][metric]

        soft_assert(
            lo_b <= baseline_acc <= hi_b,
            f"{benchmark}: baseline {baseline_acc}% out of range [{lo_b}%, {hi_b}%] for metric {metric}",
        )
        soft_assert(
            lo_a <= after_acc <= hi_a,
            f"{benchmark} for {backend}: after_training {after_acc}% out of range [{lo_a}%, {hi_a}%] for metric {metric}",
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, help="Workspace directory containing eval results.")
    ap.add_argument(
        "--backend",
        type=str,
        nargs="+",
        choices=["megatron", "fsdp"],
        default=["megatron"],
    )
    args = ap.parse_args()
    for training_backend in args.backend:
        for benchmark in ("aime24", "aime25"):
            common_path = Path(args.workspace) / "evals"
            baseline_results = load_json(common_path / "baseline" / "eval-results" / benchmark / "metrics.json")
            after_training_results = load_json(
                common_path
                / "after-training-{}".format(training_backend)
                / "eval-results"
                / benchmark
                / "metrics.json"
            )
            check_results(benchmark, baseline_results, after_training_results, training_backend)

    assert_all()


if __name__ == "__main__":
    main()
