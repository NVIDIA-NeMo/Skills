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


import pytest

from nemo_skills.evaluation.metrics.code_metrics import HiLSweBenchMetrics
from nemo_skills.inference.eval.hil_bench import parse_hil_reward


def test_parse_hil_reward_no_patch():
    assert parse_hil_reward(None, patch_exists=False) == {
        "resolved": False,
        "patch_exists": False,
        "patch_successfully_applied": False,
        "reward": 0,
        "fail_to_pass_passed": 0,
        "fail_to_pass_total": 0,
    }


def test_parse_hil_reward_apply_failed():
    metrics = parse_hil_reward({"reward": 0, "resolved": 0, "apply_failed": 1}, patch_exists=True)
    assert metrics["resolved"] is False
    assert metrics["patch_successfully_applied"] is False


def test_parse_hil_reward_success():
    metrics = parse_hil_reward(
        {
            "reward": 1,
            "resolved": 1,
            "fail_to_pass_passed": 3,
            "fail_to_pass_total": 3,
        },
        patch_exists=True,
    )
    assert metrics["resolved"] is True
    assert metrics["patch_exists"] is True
    assert metrics["patch_successfully_applied"] is True
    assert metrics["reward"] == 1.0
    assert metrics["fail_to_pass_passed"] == 3


def test_parse_hil_reward_ask_human_fields():
    metrics = parse_hil_reward(
        {
            "reward": 1,
            "resolved": 1,
            "fail_to_pass_passed": 1,
            "fail_to_pass_total": 1,
            "precision": 0.5,
            "recall": 1.0,
            "f1": 0.666,
        },
        patch_exists=True,
    )
    assert metrics["ask_f1"] == 0.666
    assert metrics["precision"] == 0.5


def test_parse_hil_reward_crash_sentinel():
    metrics = parse_hil_reward({"reward": -1}, patch_exists=True)
    assert metrics["resolved"] is False
    assert metrics["patch_successfully_applied"] is False
    assert metrics["verifier_crashed"] is True


def test_hil_swe_bench_metrics():
    metrics = HiLSweBenchMetrics()
    score = metrics._get_score_dict(
        {
            "swe-bench-metrics": {
                "resolved": True,
                "patch_exists": True,
                "patch_successfully_applied": True,
                "reward": 1,
                "ask_f1": 0.8,
                "precision": 0.75,
                "recall": 0.85,
            }
        }
    )
    assert score["issues_resolved"] is True
    assert score["reward"] == 1.0
    assert score["ask_f1"] == 0.8


def test_resolve_tests_dir_prefers_tasks_dir(tmp_path):
    from nemo_skills.inference.eval.hil_bench import HilBenchGenerationConfig, HilBenchGenerationTask

    instance_id = "swe_0__baseline"
    tests = tmp_path / "tasks" / instance_id / "tests"
    tests.mkdir(parents=True)
    (tests / "test.sh").write_text("#!/bin/bash\n")

    # Build a minimal task object without running full __init__ setup.
    task = object.__new__(HilBenchGenerationTask)
    task.cfg = HilBenchGenerationConfig(
        input_file=str(tmp_path / "in.jsonl"),
        output_file=str(tmp_path / "out.jsonl"),
        agent_framework="swe_agent",
        tasks_dir=str(tmp_path / "tasks"),
    )
    resolved = HilBenchGenerationTask._resolve_tests_dir(task, {"instance_id": instance_id})
    assert resolved == tests


def test_resolve_tests_dir_missing_raises(tmp_path):
    from nemo_skills.inference.eval.hil_bench import HilBenchGenerationConfig, HilBenchGenerationTask

    task = object.__new__(HilBenchGenerationTask)
    task.cfg = HilBenchGenerationConfig(
        input_file=str(tmp_path / "in.jsonl"),
        output_file=str(tmp_path / "out.jsonl"),
        agent_framework="swe_agent",
        tasks_dir=str(tmp_path / "tasks"),
        eval_config={},
    )
    with pytest.raises(ValueError, match="tests dir not found"):
        HilBenchGenerationTask._resolve_tests_dir(task, {"instance_id": "missing"})
