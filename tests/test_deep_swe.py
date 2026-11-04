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


import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_skills.evaluation.metrics.code_metrics import DeepSweMetrics
from nemo_skills.inference.eval.deepswe import (
    DeepSweGenerationTask,
    build_verifier_command,
    parse_deepswe_reward,
)


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


def test_parse_deepswe_reward_missing_marks_verifier_crash():
    metrics = parse_deepswe_reward(None, patch_exists=True)
    assert metrics["resolved"] is False
    assert metrics["patch_successfully_applied"] is False
    assert metrics["verifier_crashed"] is True


def test_verifier_command_uses_v11_artifact_contract():
    command = build_verifier_command("task-a")
    assert "cp /patch_mount/model.patch /logs/artifacts/model.patch" in command
    assert "export TESTS_DIR=/tests" in command
    assert "export VERIFIER_DIR=/logs/verifier" in command
    assert "export APP_DIR=/app" in command
    assert "export ARTIFACTS_DIR=/logs/artifacts" in command
    assert "bash /tests/test.sh" in command
    assert "reward.json" in command
    assert "reward.txt" in command


def test_deepswe_metrics_aggregate():
    metrics = DeepSweMetrics()
    metrics.update(
        [
            {
                "swe-bench-metrics": {
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
                "swe-bench-metrics": {
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
            "swe-bench-metrics": {
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


def test_deepswe_metrics_separates_verifier_crash_from_apply_failure():
    score = DeepSweMetrics()._get_score_dict(
        {
            "swe-bench-metrics": {
                "resolved": False,
                "patch_exists": True,
                "patch_successfully_applied": False,
                "verifier_crashed": True,
                "reward": -1,
            }
        }
    )
    assert score["verifier_crashed"] is True
    assert score["patch_cant_apply"] is False


def test_deepswe_metrics_none_when_evaluate_false():
    score = DeepSweMetrics()._get_score_dict(
        {
            "swe-bench-metrics": {
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


def _write_v11_task(task_dir: Path):
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True)
    for filename in ("test.sh", "grader.py", "config.json", "test.patch", "Dockerfile"):
        (tests_dir / filename).write_text("")
    (task_dir / "instruction.md").write_text("Fix the issue.\n")
    (task_dir / "task.toml").write_text(
        """
schema_version = "1.3"
artifacts = ["/logs/artifacts/model.patch"]
[metadata]
task_id = "task-a"
repository_url = "https://github.com/example/repo"
base_commit_hash = "abc123"
language = "python"
[agent]
timeout_sec = 10800
[verifier]
environment_mode = "separate"
timeout_sec = 1800
[[verifier.collect]]
command = "cd /app && git diff --binary abc123 HEAD > /logs/artifacts/model.patch"
[environment]
docker_image = "example/image:task-a-v1.1"
""".strip()
    )


def test_prepare_loads_v11_task_metadata(tmp_path):
    prepare = importlib.import_module("nemo_skills.dataset.deep-swe.prepare")
    task_dir = tmp_path / "task-a"
    _write_v11_task(task_dir)

    row = prepare._load_task(
        task_dir,
        source_repo="https://example.test/deep-swe.git",
        source_commit="deadbeef",
    )

    assert row["instance_id"] == "task-a"
    assert row["base_commit"] == "abc123"
    assert row["docker_image"].endswith("-v1.1")
    assert "verifier_docker_image" not in row
    assert row["agent_timeout_sec"] == 10800.0
    assert row["verifier_timeout_sec"] == 1800.0
    assert row["source_commit"] == "deadbeef"
    assert row["patch_collection"] == "base_commit_to_head"
    assert "container_formatter" not in row


def test_prepare_rejects_v10_task_layout(tmp_path):
    prepare = importlib.import_module("nemo_skills.dataset.deep-swe.prepare")
    task_dir = tmp_path / "task-a"
    _write_v11_task(task_dir)
    task_toml = task_dir / "task.toml"
    task_toml.write_text(task_toml.read_text().replace('schema_version = "1.3"', 'schema_version = "1.1"'))

    with pytest.raises(ValueError, match="not DeepSWE v1.1"):
        prepare._load_task(task_dir)


def test_deepswe_rejects_apptainer_backend():
    with pytest.raises(ValueError, match="requires .*execution_backend=opensandbox"):
        DeepSweGenerationTask(SimpleNamespace(execution_backend="apptainer"))


def test_deepswe_mini_swe_prompt_requires_commit():
    config = (
        Path(__file__).parents[1]
        / "nemo_skills"
        / "prompt"
        / "config"
        / "eval"
        / "deep-swe"
        / "mini-swe-agent"
        / "swebench.yaml"
    ).read_text()
    assert "Only committed changes are submitted" in config
    assert "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in config
    assert "Do NOT commit" not in config


def test_verifier_stages_pinned_tests_and_model_patch(tmp_path):
    task = object.__new__(DeepSweGenerationTask)
    task.output_dir = tmp_path / "results"
    input_file = tmp_path / "data" / "deep-swe" / "default.jsonl"
    tests_dir = input_file.parent / "tasks" / "task-a" / "tests"
    _write_v11_task(tests_dir.parent)
    task.cfg = SimpleNamespace(execution_backend="opensandbox", swebench_tests_timeout=60, input_file=str(input_file))
    task._execute_container_command = AsyncMock()

    metrics = asyncio.run(task._run_deepswe_verifier({"instance_id": "task-a"}, "diff --git a/a b/a\n"))

    mounts = task._execute_container_command.await_args.kwargs["extra_mounts"]
    assert [(str(source), target, read_only) for source, target, read_only in mounts] == [
        (str(task.output_dir / "patches" / "task-a.patch"), "/patch_mount/model.patch", True),
        (str(tests_dir), "/tests", True),
    ]
    assert metrics["verifier_crashed"] is True


def test_verifier_rejects_missing_pinned_tests_before_starting_sandbox(tmp_path):
    task = object.__new__(DeepSweGenerationTask)
    task.output_dir = tmp_path / "results"
    task.cfg = SimpleNamespace(input_file=str(tmp_path / "default.jsonl"))
    task._execute_container_command = AsyncMock()

    with pytest.raises(FileNotFoundError, match="Missing pinned DeepSWE verifier file"):
        asyncio.run(task._run_deepswe_verifier({"instance_id": "task-a"}, "patch"))

    task._execute_container_command.assert_not_awaited()
