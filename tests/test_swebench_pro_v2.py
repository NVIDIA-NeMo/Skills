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
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf

from nemo_skills.inference.eval.swebench_pro_v2 import (
    SweBenchProV2GenerationConfig,
    SweBenchProV2GenerationTask,
    _resolve_tests_dir,
    build_verifier_command,
    parse_swebench_pro_v2_reward,
)


def test_parse_reward_no_patch():
    assert parse_swebench_pro_v2_reward(None, patch_exists=False) == {
        "resolved": False,
        "patch_exists": False,
        "patch_successfully_applied": False,
        "reward": 0,
    }


def test_parse_reward_success():
    metrics = parse_swebench_pro_v2_reward({"reward": 1.0}, patch_exists=True)
    assert metrics["resolved"] is True
    assert metrics["patch_successfully_applied"] is True
    assert metrics["reward"] == 1.0


def test_parse_reward_apply_failure():
    metrics = parse_swebench_pro_v2_reward({"reward": 0, "apply_failed": 1}, patch_exists=True)
    assert metrics["resolved"] is False
    assert metrics["patch_successfully_applied"] is False


def test_parse_reward_missing_marks_verifier_crash():
    metrics = parse_swebench_pro_v2_reward(None, patch_exists=True)
    assert metrics["resolved"] is False
    assert metrics["verifier_crashed"] is True


def test_resolve_tests_dir_from_data_dir(tmp_path):
    instance_id = "instance_owner__repo-abc"
    tests_dir = tmp_path / "swe-bench-pro-v2" / "tasks" / instance_id / "tests"
    tests_dir.mkdir(parents=True)

    assert _resolve_tests_dir({"instance_id": instance_id}, {"data_dir": str(tmp_path)}) == tests_dir


def test_resolve_tests_dir_missing_raises(tmp_path):
    with pytest.raises(ValueError, match="tests dir not found"):
        _resolve_tests_dir({"instance_id": "missing"}, {"data_dir": str(tmp_path)})


def test_verifier_command_uses_official_patch_replay_order():
    command = build_verifier_command("instance_owner__repo-abc")
    assert "for candidate in /app /testbed" in command
    assert command.index("git apply --verbose") < command.index("git apply --3way")
    assert command.index("git apply --3way") < command.index("patch --fuzz=3")
    assert command.index("model.patch") < command.index("bash /tests/test.sh")
    assert "apply_failed" in command


def test_opensandbox_verifier_runtime_error_keeps_patch_metrics(tmp_path):
    instance_id = "instance_owner__repo-abc"
    tests_dir = tmp_path / "tasks" / instance_id / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test.sh").write_text("#!/bin/bash\n")

    task = object.__new__(SweBenchProV2GenerationTask)
    task.output_dir = tmp_path / "results"
    task.cfg = SimpleNamespace(
        tasks_dir=str(tmp_path / "tasks"),
        eval_config={},
        swebench_tests_timeout=60,
    )
    task._execute_container_command = AsyncMock(side_effect=RuntimeError("sandbox unavailable"))

    metrics = asyncio.run(task._run_v2_verifier({"instance_id": instance_id}, "diff --git a/a b/a\n"))

    assert metrics["resolved"] is False
    assert metrics["patch_exists"] is True
    assert metrics["patch_successfully_applied"] is False
    assert metrics["verifier_crashed"] is True


def test_mini_swe_agent_timeout_command_preserves_partial_patch(tmp_path):
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n")
    output_file = tmp_path / "results" / "output.jsonl"
    cfg = SweBenchProV2GenerationConfig(
        _init_nested=True,
        input_file=str(dataset),
        output_file=str(output_file),
        agent_framework="mini_swe_agent",
        server=OmegaConf.create({"model": "served-model", "base_url": "http://server/v1"}),
    )
    cfg.inference.extra_body = OmegaConf.create({})
    task = object.__new__(SweBenchProV2GenerationTask)
    task.cfg = cfg
    task.output_dir = output_file.parent
    task.output_dir.mkdir(parents=True)
    task.scratch_dir = tmp_path / "scratch"
    task.scratch_dir.mkdir()
    task.input_dir = task.scratch_dir / "input"
    task._scratch_artifacts = {}
    task.api_base = "http://server/v1"
    captured = {}

    async def fake_execute(data_point, command_builder, expected_file_pattern, **kwargs):
        captured["command"] = command_builder("http://proxy/v1")
        trajectory = tmp_path / "results" / "trajectories" / f"{data_point['instance_id']}.traj.json"
        trajectory.parent.mkdir(parents=True)
        trajectory.write_text(json.dumps({"info": {"submission": "diff --git a/a b/a\n"}}))
        return str(trajectory)

    task._execute_agent_command_with_capture = fake_execute
    prediction = asyncio.run(
        task._run_mini_swe_agent(
            {
                "instance_id": "instance_owner__repo-abc",
                "problem_statement": "Fix it",
                "agent_timeout_sec": 3000,
            }
        )
    )

    assert "timeout -k 30s 3000s" in captured["command"]
    assert "AGENT_EXIT_CODE" in captured["command"]
    assert "git diff --binary --cached" in captured["command"]
    assert json.loads(open(prediction).read())["model_patch"] == "diff --git a/a b/a\n"


def test_prepare_loads_harbor_task_metadata(tmp_path):
    prepare = importlib.import_module("nemo_skills.dataset.swe-bench-pro-v2.prepare")
    source_tasks = tmp_path / "source"
    task_dir = source_tasks / "instance_owner__repo-abc"
    tests_dir = task_dir / "tests"
    tests_dir.mkdir(parents=True)
    for filename in ("test.sh", "run_script.sh", "parser.py", "config.json", "test_patch.patch"):
        (tests_dir / filename).write_text("")
    (task_dir / "task.toml").write_text(
        """
[agent]
timeout_sec = 3000
[verifier]
timeout_sec = 3000
[environment]
docker_image = "ghcr.io/scaleapi/swe-bench_pro-v2:instance_owner__repo-abc"
""".strip()
    )

    synced_tasks = prepare._sync_tasks(source_tasks, tmp_path / "dataset" / "tasks")
    metadata = prepare._load_task_metadata(synced_tasks)

    assert metadata["instance_owner__repo-abc"] == {
        "docker_image": "ghcr.io/scaleapi/swe-bench_pro-v2:instance_owner__repo-abc",
        "agent_timeout_sec": 3000.0,
        "verifier_timeout_sec": 3000.0,
    }
