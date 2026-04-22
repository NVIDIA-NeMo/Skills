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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, help="Workspace directory containing eval results")
    args = ap.parse_args()

    for backend in METRIC_RANGES:
        check_results(args.workspace, backend)
    assert_all()


if __name__ == "__main__":
    main()
