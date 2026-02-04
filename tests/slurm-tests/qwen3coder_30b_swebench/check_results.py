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
from utils import assert_all, get_nested_value, load_json, soft_assert  # noqa: E402

METRIC_RANGES = {
    # +/- 4 pts from scores measured on 2026-01-30 (avg of 3 runs)
    "openhands": {
        ("swe-bench", "pass@1", "issues_resolved"): (44.3, 52.3),
    },
    "swe_agent": {
        ("swe-bench", "pass@1", "issues_resolved"): (45.5, 53.5),
    },
}


def check_results(eval_dir: str, agent_framework: str):
    f = os.path.join(eval_dir, "eval-results", "swe-bench", "metrics.json")
    data = load_json(f)
    for category_tuple, expected_range in METRIC_RANGES[agent_framework].items():
        val = float(get_nested_value(data, category_tuple))
        lo, hi = expected_range
        soft_assert(lo <= val <= hi, f"swe-bench ({agent_framework}) {category_tuple}={val} out of range [{lo},{hi}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, help="Workspace directory containing eval results")
    ap.add_argument("--agent_framework", required=True, help="Agent framework used for the run")
    args = ap.parse_args()

    check_results(args.workspace, args.agent_framework)

    assert_all()


if __name__ == "__main__":
    main()
