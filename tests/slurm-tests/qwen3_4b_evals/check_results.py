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
    ("overall_non_live", "accuracy"): (84.0, 90.0),
    ("non_live_ast", "accuracy"): (84.0, 92.0),
    ("non_live_irrelevance", "accuracy"): (79.0, 86.0),
    ("overall_live", "accuracy"): (76.0, 83.0),
    ("live_ast", "accuracy"): (79.0, 86.0),
    ("live_irrelevance", "accuracy"): (73.0, 80.0),
    ("live_relevance", "accuracy"): (70.0, 100.0),  # unusually high variance
    ("overall_multi_turn", "accuracy"): (20.0, 33.0),
}


GENSELECT_METRIC_RANGES = {
    "aime24": {
        ("pass@1", "symbolic_correct"): (70.0, 95.0),
    }
}


def check_results(eval_dir: str):
    f = os.path.join(eval_dir, "eval-results", "bfcl_v3", "metrics.json")
    data = load_json(f)["bfcl_v3"]
    for category_tuple, expected_range in TOOLCALLING_METRIC_RANGES.items():
        val = float(get_nested_value(data, category_tuple))
        lo, hi = expected_range
        soft_assert(lo <= val <= hi, f"bfcl-v3 {category_tuple}={val} out of range [{lo},{hi}]")

    # GenSelect Tests
    online_genselect_f = os.path.join(eval_dir, "online_genselect", "eval-results", "aime24", "metrics.json")
    online_genselect_data = load_json(online_genselect_f)["aime24"]
    for metric, (lo, hi) in GENSELECT_METRIC_RANGES["aime24"].items():
        val = float(get_nested_value(online_genselect_data, metric))
        soft_assert(lo <= val <= hi, f"online-genselect {metric}={val} out of range [{lo},{hi}]")

    # Offline GenSelect
    # 1. Check that the pass@1 score after GenSelect is within the expected range
    offline_genselect_f = os.path.join(
        eval_dir, "offline_genselect", "genselect", "eval-results", "aime24", "metrics.json"
    )
    offline_genselect_accuracy = float(load_json(offline_genselect_f)["aime24"]["pass@1"]["symbolic_correct"])

    for metric, (lo, hi) in GENSELECT_METRIC_RANGES["aime24"].items():
        val = offline_genselect_accuracy
        soft_assert(lo <= val <= hi, f"offline-genselect {metric}={val} out of range [{lo},{hi}]")

    # 2. Check that the pass@1 score after GenSelect is strictly better than the pass@1 score
    # after initial solutions (atleast 1 point better, in reality it would be much more than 1 point better)
    offline_initial_solutions_f = os.path.join(
        eval_dir, "offline_genselect", "initial_solutions", "eval-results", "aime24", "metrics.json"
    )
    offline_initial_solutions_accuracy = float(
        load_json(offline_initial_solutions_f)["aime24"]["pass@1[avg-of-8]"]["symbolic_correct"]
    )

    assert offline_genselect_accuracy >= (offline_initial_solutions_accuracy + 1.0), (
        f"Offline GenSelect pass@1 score {offline_genselect_accuracy} is not better than the initial "
        f"solutions pass@1 score {offline_initial_solutions_accuracy} by at least 1 point"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, help="Workspace directory containing eval results")
    args = ap.parse_args()

    check_results(args.workspace)

    assert_all()


if __name__ == "__main__":
    main()
