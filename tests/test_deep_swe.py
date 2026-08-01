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

from nemo_skills.evaluation.metrics.code_metrics import DeepSweMetrics
from nemo_skills.inference.eval.deepswe import _resolve_tests_dir, parse_deepswe_reward


def test_parse_deepswe_reward_no_patch():
    assert parse_deepswe_reward(None, patch_exists=False) == {
        "resolved": False,
        "patch_exists": False,
        "patch_successfully_applied": False,
        "reward": 0,
        "f2p": 0.0,
        "p2p": 0.0,
        "partial": 0.0,
    }


def test_parse_deepswe_reward_apply_failed():
    metrics = parse_deepswe_reward({"reward": 0, "apply_failed": 1, "f2p": 0.0, "p2p": 1.0}, patch_exists=True)
    assert metrics["resolved"] is False
    assert metrics["patch_successfully_applied"] is False


def test_parse_deepswe_reward_success():
    metrics = parse_deepswe_reward(
        {
            "reward": 1,
            "f2p": 1.0,
            "p2p": 1.0,
            "partial": 1.0,
            "f2p_passed": 3,
            "f2p_total": 3,
            "p2p_passed": 2,
            "p2p_total": 2,
        },
        patch_exists=True,
    )
    assert metrics["resolved"] is True
    assert metrics["patch_exists"] is True
    assert metrics["patch_successfully_applied"] is True
    assert metrics["reward"] == 1.0
    assert metrics["f2p"] == 1.0


def test_parse_deepswe_reward_crash_sentinel():
    """DeepSWE test.sh EXIT trap writes reward.txt=-1 when grading never produced reward.json."""
    metrics = parse_deepswe_reward({"reward": -1}, patch_exists=True)
    assert metrics["resolved"] is False
    assert metrics["patch_successfully_applied"] is False
    assert metrics["reward"] == -1.0
    assert metrics["verifier_crashed"] is True


def test_deepswe_metrics_aggregate():
    metrics = DeepSweMetrics()
    metrics.update(
        [
            {
                "deep-swe-metrics": {
                    "resolved": True,
                    "patch_exists": True,
                    "patch_successfully_applied": True,
                    "reward": 1,
                    "f2p": 1.0,
                    "p2p": 1.0,
                    "partial": 1.0,
                }
            }
        ]
    )
    metrics.update(
        [
            {
                "deep-swe-metrics": {
                    "resolved": False,
                    "patch_exists": True,
                    "patch_successfully_applied": True,
                    "reward": 0,
                    "f2p": 0.5,
                    "p2p": 1.0,
                    "partial": 0.75,
                }
            }
        ]
    )
    # BaseMetrics stores means over updates; spot-check resolved rate semantics via score dict helper.
    score = metrics._get_score_dict(
        {
            "deep-swe-metrics": {
                "resolved": True,
                "patch_exists": True,
                "patch_successfully_applied": True,
                "reward": 1,
                "f2p": 1.0,
                "p2p": 1.0,
                "partial": 1.0,
            }
        }
    )
    assert score["issues_resolved"] is True
    assert score["reward"] == 1.0


def test_deepswe_metrics_none_when_evaluate_false():
    score = DeepSweMetrics()._get_score_dict(
        {
            "deep-swe-metrics": {
                "resolved": None,
                "patch_exists": True,
                "patch_successfully_applied": None,
                "reward": None,
                "f2p": None,
                "p2p": None,
                "partial": None,
            }
        }
    )
    assert score["issues_resolved"] is False
    assert score["no_patch"] is False
    assert score["patch_cant_apply"] is False


def test_resolve_tests_dir_prefers_data_dir(tmp_path):
    instance_id = "abs-module-cache-flags"
    tests = tmp_path / "deep-swe" / "tasks" / instance_id / "tests"
    tests.mkdir(parents=True)
    (tests / "test.sh").write_text("#!/bin/bash\n")

    resolved = _resolve_tests_dir(
        {"instance_id": instance_id, "tests_dir": "/nonexistent/stale/path"},
        {"data_dir": str(tmp_path)},
    )
    assert resolved == tests


def test_resolve_tests_dir_explicit_test_dir_wins(tmp_path):
    instance_id = "task-a"
    via_test_dir = tmp_path / "custom_tasks" / instance_id / "tests"
    via_test_dir.mkdir(parents=True)
    via_data_dir = tmp_path / "deep-swe" / "tasks" / instance_id / "tests"
    via_data_dir.mkdir(parents=True)

    resolved = _resolve_tests_dir(
        {"instance_id": instance_id},
        {"data_dir": str(tmp_path), "test_dir": str(tmp_path / "custom_tasks")},
    )
    assert resolved == via_test_dir


def test_resolve_tests_dir_missing_raises(tmp_path):
    with pytest.raises(ValueError, match="tests dir not found"):
        _resolve_tests_dir(
            {"instance_id": "missing-task"},
            {"data_dir": str(tmp_path)},
        )
