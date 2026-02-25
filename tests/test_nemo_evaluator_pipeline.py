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

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nemo_skills.pipeline.nemo_evaluator import (
    EvaluatorClientScript,
)
from nemo_skills.pipeline.nemo_evaluator import (
    nemo_evaluator as nemo_evaluator_fn,
)
from nemo_skills.pipeline.utils.declarative import Command, CommandGroup
from nemo_skills.pipeline.utils.scripts import ServerScript


@pytest.fixture
def real_evaluator_config(monkeypatch):
    """Return path to real evaluator config file."""
    monkeypatch.setenv("NGC_API_TOKEN", "test_value_ngc_api_token")
    config_path = Path(__file__).parent / "data" / "nemo_evaluator" / "example-eval-config.yaml"
    return str(config_path)


class Ctx:
    """Mock Typer context."""

    def __init__(self, args=None):
        self.args = args or []


def _create_base_kwargs(tmp_path, real_evaluator_config, **overrides):
    """Create base kwargs for nemo_evaluator function."""
    defaults = {
        "ctx": Ctx(),
        "cluster": None,
        "output_dir": str(tmp_path / "out"),
        "expname": "evaluator-test",
        "nemo_evaluator_config": real_evaluator_config,
        "job_gpus": 0,
        "job_nodes": 1,
        "partition": None,
        "qos": None,
        "mount_paths": None,
        "log_dir": None,
        "exclusive": False,
        "with_sandbox": False,
        "keep_mounts_for_sandbox": False,
        "server_type": None,
        "server_model": None,
        "server_gpus": 0,
        "server_nodes": 1,
        "server_port": None,
        "server_args": None,
        "server_entrypoint": None,
        "server_container": None,
        "server_base_url": None,
        "server_api_path": "/v1/chat/completions",
        "server_health_path": "/health",
        "judge_server_type": None,
        "judge_server_model": None,
        "judge_server_gpus": 0,
        "judge_server_nodes": 1,
        "judge_server_port": None,
        "judge_server_args": None,
        "judge_server_entrypoint": None,
        "judge_server_container": None,
        "judge_server_base_url": None,
        "judge_server_api_path": "/v1/chat/completions",
        "judge_server_health_path": "/health",
        "reuse_code": True,
        "reuse_code_exp": None,
        "run_after": None,
        "dependent_jobs": 0,
        "config_dir": None,
        "dry_run": True,
    }
    defaults.update(overrides)
    return defaults


@patch("nemo_skills.pipeline.nemo_evaluator.Pipeline")
def test_no_servers_external_urls(
    mock_pipeline_cls,
    tmp_path,
    real_evaluator_config,
):
    """Test path: No hosted servers, using external URLs."""
    mock_pipeline_instance = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline_instance

    kwargs = _create_base_kwargs(
        tmp_path,
        real_evaluator_config,
        server_base_url="http://external-server:8000",
        judge_server_base_url="http://external-judge:9000",
    )

    # Call function
    nemo_evaluator_fn(**kwargs)

    # Verify Pipeline was created correctly
    assert mock_pipeline_cls.called
    call_kwargs = mock_pipeline_cls.call_args[1]
    assert call_kwargs["name"] == "evaluator-test"
    assert call_kwargs["reuse_code"] is True
    assert call_kwargs["skip_hf_home_check"] is True
    assert call_kwargs["with_ray"] is False

    # Verify jobs structure (real config has 2 tasks: ifeval and gpqa_diamond)
    jobs = call_kwargs["jobs"]
    assert len(jobs) == 2
    job = jobs[0]
    assert job["name"] == "evaluator-test-0"
    assert "groups" in job
    assert len(job["groups"]) == 1

    # Verify single group structure
    group = job["groups"][0]
    assert isinstance(group, CommandGroup)
    assert group.name == "evaluator-test-0"
    assert len(group.commands) == 1  # Only client command

    # Verify client command
    client_cmd = group.commands[0]
    assert isinstance(client_cmd, Command)
    assert client_cmd.name.startswith("evaluator-test-client-0")
    assert isinstance(client_cmd.script, EvaluatorClientScript)

    # Verify hardware config
    assert group.hardware is not None
    assert group.hardware.num_gpus is None  # job_gpus=0 means None
    assert group.hardware.num_nodes == 1


@patch("nemo_skills.pipeline.nemo_evaluator.Pipeline")
def test_main_server_hosted(
    mock_pipeline_cls,
    tmp_path,
    real_evaluator_config,
):
    """Test path: Main server hosted, no judge server."""
    mock_pipeline_instance = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline_instance

    kwargs = _create_base_kwargs(
        tmp_path,
        real_evaluator_config,
        server_type="vllm",
        server_model="meta/llama-3.1-8b-instruct",
        server_gpus=8,
        server_nodes=1,
    )

    # Call function
    nemo_evaluator_fn(**kwargs)

    # Verify Pipeline structure (real config has 2 tasks)
    call_kwargs = mock_pipeline_cls.call_args[1]
    jobs = call_kwargs["jobs"]
    assert len(jobs) == 2
    job = jobs[0]
    assert job["name"] == "evaluator-test-0"
    assert "groups" in job
    assert len(job["groups"]) == 1

    # Verify single group with server + client
    group = job["groups"][0]
    assert isinstance(group, CommandGroup)
    assert len(group.commands) == 2  # Server + client

    # Verify server command
    server_cmd = group.commands[0]
    assert isinstance(server_cmd, Command)
    assert "server" in server_cmd.name
    assert isinstance(server_cmd.script, ServerScript)
    assert server_cmd.script.num_gpus == 8
    assert server_cmd.script.log_prefix == "server"
    assert server_cmd.script.port is not None

    # Verify client command
    client_cmd = group.commands[1]
    assert isinstance(client_cmd, Command)
    assert "client" in client_cmd.name
    assert isinstance(client_cmd.script, EvaluatorClientScript)
    assert callable(client_cmd.script.inline)  # Should be lambda for cross-component refs

    # Verify hardware config (should use server GPUs)
    assert group.hardware.num_gpus == 8
    assert group.hardware.num_nodes == 1


@patch("nemo_skills.pipeline.nemo_evaluator.Pipeline")
def test_judge_server_hosted(
    mock_pipeline_cls,
    tmp_path,
    real_evaluator_config,
):
    """Test path: Judge server hosted, no main server."""
    mock_pipeline_instance = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline_instance

    kwargs = _create_base_kwargs(
        tmp_path,
        real_evaluator_config,
        judge_server_type="vllm",
        judge_server_model="meta/llama-3.1-32b-instruct",
        judge_server_gpus=32,
        judge_server_nodes=1,
    )

    # Call function
    nemo_evaluator_fn(**kwargs)

    # Verify Pipeline structure (real config has 2 tasks)
    call_kwargs = mock_pipeline_cls.call_args[1]
    jobs = call_kwargs["jobs"]
    assert len(jobs) == 2
    job = jobs[0]
    assert "groups" in job
    assert len(job["groups"]) == 1

    # Verify single group with judge server + client
    group = job["groups"][0]
    assert len(group.commands) == 2  # Judge server + client

    # Verify judge server command
    judge_cmd = group.commands[0]
    assert isinstance(judge_cmd, Command)
    assert "judge-server" in judge_cmd.name
    assert isinstance(judge_cmd.script, ServerScript)
    assert judge_cmd.script.num_gpus == 32
    assert judge_cmd.script.log_prefix == "judge-server"

    # Verify client command
    client_cmd = group.commands[1]
    assert isinstance(client_cmd, Command)
    assert "client" in client_cmd.name
    assert isinstance(client_cmd.script, EvaluatorClientScript)
    assert callable(client_cmd.script.inline)  # Should be lambda for cross-component refs

    # Verify hardware config (should use judge server GPUs)
    assert group.hardware.num_gpus == 32


@patch("nemo_skills.pipeline.nemo_evaluator.Pipeline")
def test_both_servers_hosted_separate_groups(
    mock_pipeline_cls,
    tmp_path,
    real_evaluator_config,
):
    """Test path: Both main and judge servers hosted - should create separate groups."""
    mock_pipeline_instance = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline_instance

    kwargs = _create_base_kwargs(
        tmp_path,
        real_evaluator_config,
        server_type="vllm",
        server_model="meta/llama-3.1-8b-instruct",
        server_gpus=8,
        judge_server_type="vllm",
        judge_server_model="meta/llama-3.1-32b-instruct",
        judge_server_gpus=32,
    )

    # Call function
    nemo_evaluator_fn(**kwargs)

    # Verify Pipeline structure (real config has 2 tasks, so 2 jobs)
    call_kwargs = mock_pipeline_cls.call_args[1]
    jobs = call_kwargs["jobs"]
    assert len(jobs) == 2
    job = jobs[0]
    assert "groups" in job
    assert len(job["groups"]) == 2  # Two separate groups

    # Verify first group: main server + client
    server_group = job["groups"][0]
    assert isinstance(server_group, CommandGroup)
    assert "server" in server_group.name
    assert len(server_group.commands) == 2  # Server + client
    assert server_group.hardware.num_gpus == 8
    assert server_group.hardware.num_nodes == 1

    # Verify second group: judge server only
    judge_group = job["groups"][1]
    assert isinstance(judge_group, CommandGroup)
    assert "judge-server" in judge_group.name
    assert len(judge_group.commands) == 1  # Only judge server
    assert judge_group.hardware.num_gpus == 32
    assert judge_group.hardware.num_nodes == 1

    # Verify server command in first group
    server_cmd = server_group.commands[0]
    assert isinstance(server_cmd, Command)
    assert "server" in server_cmd.name
    assert isinstance(server_cmd.script, ServerScript)
    assert server_cmd.script.num_gpus == 8

    # Verify client command in first group
    client_cmd = server_group.commands[1]
    assert isinstance(client_cmd, Command)
    assert "client" in client_cmd.name
    assert isinstance(client_cmd.script, EvaluatorClientScript)
    assert callable(client_cmd.script.inline)  # Lambda for cross-component refs

    # Verify judge server command in second group
    judge_cmd = judge_group.commands[0]
    assert isinstance(judge_cmd, Command)
    assert "judge-server" in judge_cmd.name
    assert isinstance(judge_cmd.script, ServerScript)
    assert judge_cmd.script.num_gpus == 32


@patch("nemo_skills.pipeline.nemo_evaluator.Pipeline")
def test_multiple_tasks(
    mock_pipeline_cls,
    tmp_path,
    real_evaluator_config,
):
    """Test path: Multiple tasks create multiple jobs (real config has 2 tasks)."""
    mock_pipeline_instance = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline_instance

    kwargs = _create_base_kwargs(tmp_path, real_evaluator_config)

    # Call function
    nemo_evaluator_fn(**kwargs)

    # Verify Pipeline structure (real config has 2 tasks: ifeval and gpqa_diamond)
    call_kwargs = mock_pipeline_cls.call_args[1]
    jobs = call_kwargs["jobs"]
    assert len(jobs) == 2  # One job per task

    # Verify first job
    job0 = jobs[0]
    assert job0["name"] == "evaluator-test-0"
    assert len(job0["groups"]) == 1

    # Verify second job
    job1 = jobs[1]
    assert job1["name"] == "evaluator-test-1"
    assert len(job1["groups"]) == 1


@patch("nemo_skills.pipeline.nemo_evaluator.Pipeline")
def test_output_dir_structure(
    mock_pipeline_cls,
    tmp_path,
    real_evaluator_config,
):
    """Test that output directory structure is correctly set in task config."""
    mock_pipeline_instance = MagicMock()
    mock_pipeline_cls.return_value = mock_pipeline_instance

    output_dir = str(tmp_path / "out")
    kwargs = _create_base_kwargs(tmp_path, real_evaluator_config, output_dir=output_dir, expname="test-exp")

    # Call function
    nemo_evaluator_fn(**kwargs)

    # Verify Pipeline was created
    assert mock_pipeline_cls.called
    call_kwargs = mock_pipeline_cls.call_args[1]
    jobs = call_kwargs["jobs"]
    assert len(jobs) == 2  # Real config has 2 tasks
