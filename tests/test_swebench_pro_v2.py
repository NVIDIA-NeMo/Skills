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
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_skills.dataset.swe_bench_pro_utils import ALPINE_INSTANCE_IDS
from nemo_skills.inference.eval.swebench import SweBenchGenerationConfig, SweBenchGenerationTask
from nemo_skills.inference.eval.swebench_pro_v2 import (
    SweBenchProV2GenerationTask,
    _resolve_tests_dir,
    parse_swebench_pro_v2_reward,
)

prepare = importlib.import_module("nemo_skills.dataset.swe-bench-pro-v2.prepare")


def test_parse_reward_states():
    assert parse_swebench_pro_v2_reward(None, patch_exists=False) == {
        "resolved": False,
        "patch_exists": False,
        "patch_successfully_applied": False,
        "reward": 0,
    }
    failed = parse_swebench_pro_v2_reward(
        {"reward": 0, "apply_failed": 1, "apply_error": "bad diff"},
        patch_exists=True,
    )
    assert failed["patch_successfully_applied"] is False
    assert failed["apply_error"] == "bad diff"
    assert parse_swebench_pro_v2_reward({"reward": 1}, patch_exists=True)["resolved"] is True
    assert parse_swebench_pro_v2_reward({"reward": 0}, patch_exists=True)["resolved"] is False


def test_load_task_uses_harbor_instruction_and_gold_patch(tmp_path):
    task = tmp_path / "instance_example__repo-abc"
    (task / "tests").mkdir(parents=True)
    (task / "solution").mkdir()
    (task / "instruction.md").write_text("Fix the V2 task.\n")
    (task / "tests" / "test.sh").write_text("#!/bin/bash\n")
    (task / "solution" / "gold_patch.diff").write_text("diff --git a/a b/a\n")
    (task / "task.toml").write_text(
        """
[metadata]
category = "debugging"
difficulty = "hard"
[agent]
timeout_sec = 3000
[verifier]
timeout_sec = 3000
[environment]
docker_image = "ghcr.io/scaleapi/swe-bench_pro-v2:instance_example__repo-abc"
"""
    )
    row = {
        "instance_id": task.name,
        "repo": "example/repo",
        "base_commit": "abc",
        "repo_language": "js",
        "version": "2.0.0",
        "hard": True,
    }

    loaded = prepare._load_task(task, row, "docker://{docker_image}", "ScaleAI/SWE-bench_Pro")

    assert loaded["problem_statement"] == "Fix the V2 task.\n"
    assert loaded["patch"] == "diff --git a/a b/a\n"
    assert loaded["language"] == "javascript"
    assert loaded["container_repo_dir_fallback"] == "/testbed"
    assert loaded["container_formatter"].startswith("docker://ghcr.io/scaleapi/")


def test_resolve_tests_dir_from_data_dir(tmp_path):
    instance_id = "instance_example__repo-abc"
    tests = tmp_path / "swe-bench-pro-v2" / "tasks" / instance_id / "tests"
    tests.mkdir(parents=True)
    (tests / "test.sh").write_text("#!/bin/bash\n")
    assert _resolve_tests_dir({"instance_id": instance_id}, {"data_dir": str(tmp_path)}) == tests


def test_locked_network_relay_command_is_opt_in():
    required = {
        "input_file": "input.jsonl",
        "output_file": "output.jsonl",
        "agent_framework": "mini_swe_agent",
    }
    task = object.__new__(SweBenchGenerationTask)
    task.cfg = SweBenchGenerationConfig(**required)
    assert task._get_agent_relay_command("python") == ""

    task.cfg = SweBenchGenerationConfig(**required, isolate_agent_network=True)
    command = task._get_agent_relay_command("python")
    assert "/llm_proxy/unix_socket_relay.py" in command
    assert "--socket /llm_proxy/proxy.sock" in command


def test_v2_verifier_applies_patch_before_tests(tmp_path):
    async def run_test():
        instance_id = "instance_example__repo-abc"
        tests = tmp_path / "tasks" / instance_id / "tests"
        tests.mkdir(parents=True)
        (tests / "test.sh").write_text("#!/bin/bash\n")

        task = object.__new__(SweBenchProV2GenerationTask)
        task.output_dir = tmp_path / "output"
        task.cfg = SimpleNamespace(
            eval_config={},
            tasks_dir=str(tmp_path / "tasks"),
            swebench_tests_timeout=30,
            use_agent_timeouts=True,
            use_verifier_timeouts=True,
        )
        captured = {}

        async def execute(data_point, command, expected_file_pattern, **kwargs):
            captured["command"] = command
            captured["kwargs"] = kwargs
            eval_out = task.output_dir / "eval-outputs" / instance_id
            (eval_out / "reward.txt").write_text("1\n")
            return str(eval_out / "reward.txt")

        task._execute_container_command = execute
        metrics = await task._run_swebench_pro_v2_verifier(
            {"instance_id": instance_id},
            "diff --git a/a b/a\n",
        )
        assert metrics["resolved"] is True
        assert "repo=/app; [ -d /app/.git ] || repo=/testbed" in captured["command"]
        assert "git apply --verbose" in captured["command"]
        assert "git apply --3way" in captured["command"]
        assert "bash /tests/test.sh" in captured["command"]

    asyncio.run(run_test())


def test_local_release_has_expected_os_split():
    repo = Path("/home/wahmad/Desktop/workspace/git_repos/SWE-bench_Pro-os")
    if not (repo / ".git").exists():
        pytest.skip("local SWE-bench Pro V2 checkout is unavailable")
    files = subprocess.check_output(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", "HEAD", "v2/tasks"],
        text=True,
    ).splitlines()
    task_ids = {path.split("/")[2] for path in files if path.endswith("/task.toml")}
    assert len(task_ids) == 642
    assert len(task_ids & ALPINE_INSTANCE_IDS) == 77
    assert len(task_ids - ALPINE_INSTANCE_IDS) == 565
