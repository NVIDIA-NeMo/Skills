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

REASONING_TASKS = [
    "math-500",
    "aime24",
    "aime25",
    "gpqa",
    "mmlu-pro",
    "livecodebench",
    "scicode",
    "hle",
]

REASONING_BENCHMARKS_SCIENCE_HLE = {"scicode", "hle"}

REASONING_REQUIRED_FIELDS = {
    "math-500": ["symbolic_correct"],
    "aime24": ["symbolic_correct"],
    "aime25": ["symbolic_correct"],
    "gpqa": ["symbolic_correct"],
    "mmlu-pro": ["symbolic_correct"],
    "livecodebench": ["accuracy"],
    "scicode": ["problem_accuracy", "subtask_accuracy"],
    "hle": ["judge_correct"],
}

REASONING_METRIC_RANGES = {
    "reasoning_on": {
        "math-500": (96, 99.0),
        "aime24": (84.0, 94.0),
        "aime25": (78.0, 88.0),
        "gpqa": (72.0, 77.0),
        "mmlu-pro": (80.0, 83.0),
        "livecodebench": (69.0, 74.0),
        "scicode": {
            "problem_accuracy": (11.0, 16.0),
            "subtask_accuracy": (35.0, 40.0),
        },
        "hle": {
            "judge_correct": (5.0, 10.0),
        },
    },
    "reasoning_off": {
        "math-500": (72.0, 77.0),
        "aime24": (11.0, 21.0),
        "aime25": (0.0, 10.0),
        "gpqa": (49.0, 56.0),
        "mmlu-pro": (68.0, 71.0),
        "livecodebench": (26.0, 32.5),
        "scicode": {
            "problem_accuracy": (5.0, 10.0),
            "subtask_accuracy": (20.0, 28.0),
        },
        "hle": {
            "judge_correct": (1.5, 6.5),
        },
    },
}

TOOLCALLING_METRIC_PATHS = {
    "overall_accuracy": ["overall_accuracy", "accuracy"],
    "overall_non_live": ["non_live_single_turn", "overall_non_live", "accuracy"],
    "non_live_ast": ["non_live_single_turn", "non_live_ast", "accuracy"],
    "irrelevance": ["non_live_single_turn", "irrelevance", "accuracy"],
    "overall_live": ["live_single_turn", "overall_live", "accuracy"],
    "live_ast": ["live_single_turn", "live_ast", "accuracy"],
    "live_irrelevance": ["live_single_turn", "live_irrelevance", "accuracy"],
    "live_relevance": ["live_single_turn", "live_relevance", "accuracy"],
    "overall_multi_turn": ["multi_turn", "overall_multi_turn", "accuracy"],
}

TOOLCALLING_METRIC_RANGES = {
    "reasoning_on": {
        "overall_accuracy": (70.0, 75.0),
        "overall_non_live": (85.7, 90.7),
        "non_live_ast": (86.0, 91.0),
        "irrelevance": (84.0, 89.0),
        "overall_live": (81.0, 86.0),
        "live_ast": (80.0, 85.0),
        "live_irrelevance": (82.0, 87.0),
        "live_relevance": (60.0, 80.0),  # unusually high variance
        "overall_multi_turn": (44.0, 49.0),
    },
    "reasoning_off": {
        "overall_accuracy": (66.0, 71.0),
        "overall_non_live": (85.0, 90.0),
        "non_live_ast": (85.0, 90.0),
        "irrelevance": (86.0, 91.0),
        "overall_live": (79.0, 84.0),
        "live_ast": (77.0, 82.0),
        "live_irrelevance": (83.0, 88.0),
        "live_relevance": (53.0, 58.0),
        "overall_multi_turn": (33.5, 38.5),
    },
}

RULER_TASKS = [
    "ruler.nemotron_super_128k_slurm_ci",
    "ruler.nemotron_super_128k_slurm_ci.niah_single_1",
    "ruler.nemotron_super_128k_slurm_ci.niah_single_2",
    "ruler.nemotron_super_128k_slurm_ci.niah_single_3",
    "ruler.nemotron_super_128k_slurm_ci.niah_multikey_1",
    "ruler.nemotron_super_128k_slurm_ci.niah_multikey_2",
    "ruler.nemotron_super_128k_slurm_ci.niah_multikey_3",
    "ruler.nemotron_super_128k_slurm_ci.niah_multivalue",
    "ruler.nemotron_super_128k_slurm_ci.niah_multiquery",
    "ruler.nemotron_super_128k_slurm_ci.vt",
    "ruler.nemotron_super_128k_slurm_ci.cwe",
    "ruler.nemotron_super_128k_slurm_ci.fwe",
    "ruler.nemotron_super_128k_slurm_ci.qa_1",
    "ruler.nemotron_super_128k_slurm_ci.qa_2",
]

RULER_METRIC_RANGES = {
    "reasoning_off": {
        "ruler.nemotron_super_128k_slurm_ci": (64.5, 69.5),
        "ruler.nemotron_super_128k_slurm_ci.niah_single_1": (97.5, 100.0),
        "ruler.nemotron_super_128k_slurm_ci.niah_single_2": (91.5, 96.5),
        "ruler.nemotron_super_128k_slurm_ci.niah_single_3": (97.5, 100.0),
        "ruler.nemotron_super_128k_slurm_ci.niah_multikey_1": (73.0, 79.0),
        "ruler.nemotron_super_128k_slurm_ci.niah_multikey_2": (58.0, 68.0),
        "ruler.nemotron_super_128k_slurm_ci.niah_multikey_3": (18.0, 23.0),
        "ruler.nemotron_super_128k_slurm_ci.niah_multivalue": (80.5, 86.5),
        "ruler.nemotron_super_128k_slurm_ci.niah_multiquery": (83.0, 88.0),
        "ruler.nemotron_super_128k_slurm_ci.vt": (78.0, 84.0),
        "ruler.nemotron_super_128k_slurm_ci.cwe": (0.0, 2.0),
        "ruler.nemotron_super_128k_slurm_ci.fwe": (86.0, 92.0),
        "ruler.nemotron_super_128k_slurm_ci.qa_1": (40.0, 48.0),
        "ruler.nemotron_super_128k_slurm_ci.qa_2": (35.0, 42.0),
    },
}


def check_reasoning(eval_dir: str, mode: str):
    for bench in REASONING_TASKS:
        f = os.path.join(eval_dir, "eval-results", bench, "metrics.json")
        data = load_json(f)
        if bench in {"math-500", "aime24", "aime25", "gpqa", "livecodebench", "scicode"}:
            result_block = data[bench]["pass@1[avg-of-4]"]
        elif bench in {"mmlu-pro", "hle"}:
            result_block = data[bench]["pass@1"]
        else:
            raise RuntimeError(f"Unexpected benchmark: {bench}")

        if bench in REASONING_BENCHMARKS_SCIENCE_HLE:
            for field in REASONING_REQUIRED_FIELDS[bench]:
                val = float(result_block[field])
                lo, hi = REASONING_METRIC_RANGES[mode][bench][field]
                soft_assert(lo <= val <= hi, f"{bench} ({mode}) {field}={val} out of range [{lo},{hi}]")
        else:
            field = REASONING_REQUIRED_FIELDS[bench][0]
            val = float(result_block[field])
            lo, hi = REASONING_METRIC_RANGES[mode][bench]
            soft_assert(lo <= val <= hi, f"{bench} ({mode}) {field}={val} out of range [{lo},{hi}]")


def check_ruler(eval_dir: str, mode: str):
    f = os.path.join(eval_dir, "eval-results", "ruler.nemotron_super_128k_slurm_ci", "metrics.json")
    data = load_json(f)
    for task in RULER_TASKS:
        val = float(data[task]["pass@1"]["accuracy"])
        lo, hi = RULER_METRIC_RANGES[mode][task]
        soft_assert(lo <= val <= hi, f"RULER ({mode}) {task}={val} out of range [{lo},{hi}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, help="Workspace directory containing eval results")
    args = ap.parse_args()

    eval_root = Path(args.workspace)

    check_reasoning(eval_root / "reasoning_off", "reasoning_off")
    check_reasoning(eval_root / "reasoning_on", "reasoning_on")
    check_ruler(eval_root / "reasoning_off_ruler", "reasoning_off")

    assert_all()


if __name__ == "__main__":
    main()
