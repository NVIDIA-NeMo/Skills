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
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # for utils.py
from utils import assert_all, load_json, soft_assert  # noqa: E402

RANGE_CONSTRAINTS = {
    "aime25": {"pass@1[avg-of-16]": (93.33, 100.0), "majority@16": (100.0, 100.0), "pass@16": (100.0, 100.0)},
}


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

    assert_all()


if __name__ == "__main__":
    main()
