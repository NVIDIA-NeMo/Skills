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

import json
from pathlib import Path

import pytest

from nemo_skills.evaluation.metrics.code_metrics import SeniorSweBenchMetrics
from nemo_skills.inference.eval.harbor_utils import load_verifier_reward, resolve_tests_dir
from nemo_skills.inference.eval.senior_swe_bench import parse_senior_swe_bench_reward


def test_parse_ssb_reward_no_patch():
    assert parse_senior_swe_bench_reward(None, patch_exists=False) == {
        "resolved": False,
        "patch_exists": False,
        "patch_successfully_applied": False,
        "reward": 0,
        "invalid_trial": False,
    }


def test_parse_ssb_reward_apply_failed():
    metrics = parse_senior_swe_bench_reward({"reward": 0, "apply_failed": 1}, patch_exists=True)
    assert metrics["resolved"] is False
    assert metrics["patch_successfully_applied"] is False
    assert metrics["invalid_trial"] is False


def test_parse_ssb_reward_success():
    metrics = parse_senior_swe_bench_reward(
        {
            "reward": 1,
            "verifier_score": 1.0,
            "rubric_score": 0.9,
            "validation_score": 1.0,
        },
        patch_exists=True,
    )
    assert metrics["resolved"] is True
    assert metrics["patch_exists"] is True
    assert metrics["patch_successfully_applied"] is True
    assert metrics["reward"] == 1.0
    assert metrics["verifier_score"] == 1.0
    assert metrics["invalid_trial"] is False


def test_parse_ssb_reward_invalid_trial():
    metrics = parse_senior_swe_bench_reward({"invalid_trial": True}, patch_exists=True)
    assert metrics["resolved"] is False
    assert metrics["invalid_trial"] is True
    assert metrics["reward"] == 0


def test_parse_ssb_reward_missing_is_invalid():
    metrics = parse_senior_swe_bench_reward(None, patch_exists=True)
    assert metrics["resolved"] is False
    assert metrics["invalid_trial"] is True


def test_load_verifier_reward_empty_txt_is_invalid(tmp_path: Path):
    (tmp_path / "reward.txt").write_text("")
    assert load_verifier_reward(tmp_path) == {"invalid_trial": True}


def test_load_verifier_reward_json(tmp_path: Path):
    (tmp_path / "reward.json").write_text(json.dumps({"reward": 1, "verifier_score": 1.0}))
    reward = load_verifier_reward(tmp_path)
    assert reward["reward"] == 1
    assert reward["verifier_score"] == 1.0


def test_ssb_metrics_aggregate():
    score = SeniorSweBenchMetrics()._get_score_dict(
        {
            "swe-bench-metrics": {
                "resolved": True,
                "patch_exists": True,
                "patch_successfully_applied": True,
                "reward": 1,
                "invalid_trial": False,
                "verifier_score": 1.0,
            }
        }
    )
    assert score["issues_resolved"] is True
    assert score["reward"] == 1.0
    assert score["invalid_trial"] is False


def test_ssb_metrics_none_when_evaluate_false():
    score = SeniorSweBenchMetrics()._get_score_dict(
        {
            "swe-bench-metrics": {
                "resolved": None,
                "patch_exists": True,
                "patch_successfully_applied": None,
                "reward": None,
                "invalid_trial": None,
            }
        }
    )
    assert score["issues_resolved"] is False
    assert score["no_patch"] is False
    assert score["patch_cant_apply"] is False
    assert score["invalid_trial"] is False


def test_resolve_tests_dir_prefers_data_dir(tmp_path: Path):
    instance_id = "harbor-refactor-optional-sandbox-deps"
    tests = tmp_path / "senior-swe-bench" / "tasks" / instance_id / "tests"
    tests.mkdir(parents=True)
    (tests / "test.sh").write_text("#!/bin/bash\n")

    resolved = resolve_tests_dir(
        {"instance_id": instance_id, "tests_dir": "/nonexistent/stale/path"},
        {"data_dir": str(tmp_path)},
        benchmark_name="senior-swe-bench",
    )
    assert resolved == tests


def test_resolve_tests_dir_explicit_test_dir_wins(tmp_path: Path):
    instance_id = "task-a"
    via_test_dir = tmp_path / "custom_tasks" / instance_id / "tests"
    via_test_dir.mkdir(parents=True)
    via_data_dir = tmp_path / "senior-swe-bench" / "tasks" / instance_id / "tests"
    via_data_dir.mkdir(parents=True)

    resolved = resolve_tests_dir(
        {"instance_id": instance_id},
        {"data_dir": str(tmp_path), "test_dir": str(tmp_path / "custom_tasks")},
        benchmark_name="senior-swe-bench",
    )
    assert resolved == via_test_dir


def test_resolve_tests_dir_missing_raises(tmp_path: Path):
    with pytest.raises(ValueError, match="tests dir not found"):
        resolve_tests_dir(
            {"instance_id": "missing-task"},
            {"data_dir": str(tmp_path)},
            benchmark_name="senior-swe-bench",
        )
