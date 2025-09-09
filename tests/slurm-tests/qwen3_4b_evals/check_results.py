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

TOOLCALLING_METRIC_RANGES = {
    ("overall_accuracy", "accuracy"): (61.0, 67.0),
    ("non_live_single_turn", "overall_non_live", "accuracy"): (84.0, 90.0),
    ("non_live_single_turn", "non_live_ast", "accuracy"): (85.0, 92.0),
    ("non_live_single_turn", "irrelevance", "accuracy"): (79.0, 86.0),
    ("live_single_turn", "overall_live", "accuracy"): (76.0, 83.0),
    ("live_single_turn", "live_ast", "accuracy"): (79.0, 86.0),
    ("live_single_turn", "live_irrelevance", "accuracy"): (73.0, 80.0),
    ("live_single_turn", "live_relevance", "accuracy"): (70.0, 90.0),  # unusually high variance
    ("multi_turn", "overall_multi_turn", "accuracy"): (20.0, 30.0),
}


def check_results(eval_dir: str):
    f = os.path.join(eval_dir, "eval-results", "bfcl_v3", "metrics.json")
    data = load_json(f)
    for category_tuple, expected_range in TOOLCALLING_METRIC_RANGES.items():
        val = float(get_nested_value(data, category_tuple))
        lo, hi = expected_range
        soft_assert(lo <= val <= hi, f"bfcl-v3 {category_tuple}={val} out of range [{lo},{hi}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, help="Workspace directory containing eval results")
    args = ap.parse_args()

    check_results(args.workspace)

    assert_all()


if __name__ == "__main__":
    main()
