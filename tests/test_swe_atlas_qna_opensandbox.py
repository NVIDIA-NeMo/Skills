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
import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from nemo_skills.inference.eval import opensandbox
from nemo_skills.inference.eval.first_request_proxy import capture_first_llm_request
from nemo_skills.inference.eval.opensandbox import (
    OpenSandboxCommandTimeout,
    OpenSandboxExecutor,
    _extract_outputs,
    get_task_image,
)
from nemo_skills.inference.eval.swe_atlas_qna import (
    NEMO_RUN_EVAL_DIR,
    SweAtlasQnAGenerationConfig,
    SweAtlasQnAGenerationTask,
)
from nemo_skills.inference.eval.swebench import SupportedAgentFrameworks
from nemo_skills.pipeline.eval import _uses_apptainer


def _tar_gz(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class FakeSandbox:
    def __init__(self, events, outputs, failing_command=None):
        self.events = events
        self.outputs = outputs
        self.failing_command = failing_command
        self.commands = SimpleNamespace(
            run=self._run,
            get_command_status=self._status,
            get_background_command_logs=self._logs,
        )
        self.files = SimpleNamespace(write_file=self._write_file, read_bytes=self._read_bytes)
        self._exit_codes = {}

    async def _run(self, command, opts):
        command_id = f"cmd-{len(self._exit_codes)}"
        self.events.append(("run", command))
        failed = self.failing_command is not None and self.failing_command in command
        self._exit_codes[command_id] = 1 if failed else 0
        return SimpleNamespace(id=command_id, error=None)

    async def _status(self, command_id):
        return SimpleNamespace(running=False, exit_code=self._exit_codes[command_id], error=None)

    async def _logs(self, command_id, cursor=None):
        return SimpleNamespace(content="", cursor=cursor)

    async def _write_file(self, path, data, mode):
        self.events.append(("write", path, data))

    async def _read_bytes(self, path):
        self.events.append(("read", path))
        return _tar_gz(self.outputs)

    async def delete_egress_rules(self, targets):
        self.events.append(("delete_egress", tuple(targets)))

    async def kill(self):
        self.events.append(("kill",))

    async def close(self):
        self.events.append(("close",))


def _fake_executor(monkeypatch, events, outputs, failing_command=None, **options):
    created = []

    class FakeSandboxFactory:
        @staticmethod
        async def create(**kwargs):
            created.append(kwargs)
            return FakeSandbox(events, outputs, failing_command)

    class Record:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setenv("OPENSANDBOX_DOMAIN", "https://sandbox.example.com")
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "secret")
    monkeypatch.setattr(opensandbox, "_load_sdk", lambda: (FakeSandboxFactory, Record, Record, Record, Record))
    executor = OpenSandboxExecutor(poll_interval_s=0.01, **options)
    return executor, created


def _run_agent(executor, tmp_path, **overrides):
    upload = tmp_path / "helper.py"
    upload.write_text("print('helper')", encoding="utf-8")
    kwargs = dict(
        instance_id="task/1",
        image="ghcr.io/scaleapi/swe-atlas:task",
        setup_command="install-agent",
        runtime_dirs=("mini-swe-agent", "uv"),
        uploads=[(upload, "/nemo_run/code/nemo_skills/inference/eval/helper.py")],
        repo_setup_command="cp -r /app /testbed",
        command="run-agent",
        timeout=60,
        setup_timeout=60,
        proxy_host="10.0.0.5",
        outputs_destination=tmp_path / "staged",
        log_file=tmp_path / "logs" / "task.log",
    )
    kwargs.update(overrides)
    asyncio.run(executor.run_agent(**kwargs))


def test_opensandbox_run_agent_stages_runtime_and_downloads_outputs(monkeypatch, tmp_path):
    events = []
    executor, created = _fake_executor(
        monkeypatch, events, {"./trajectories/task.traj.json": b'{"info": {}}'}, resources={"cpu": 4}
    )

    _run_agent(executor, tmp_path)

    assert created[0]["image"] == "ghcr.io/scaleapi/swe-atlas:task"
    assert created[0]["resource"] == {"cpu": "4", "memory": "8Gi"}
    assert created[0]["network_policy"] is None
    assert created[0]["metadata"]["instance_id"] == "taskx1"
    assert executor.connection_options["headers"] == {"OPEN-SANDBOX-API-KEY": "secret"}

    runs = [event[1] for event in events if event[0] == "run"]
    write_index = next(index for index, event in enumerate(events) if event[0] == "write")
    setup_index = next(
        index for index, event in enumerate(events) if event[0] == "run" and "install-agent" in event[1]
    )
    assert write_index < setup_index, "helper scripts must be uploaded before setup runs them"
    setup = next(command for command in runs if "install-agent" in command)
    assert "apk add --no-cache git curl ca-certificates make" in setup
    assert "mv /root/mini-swe-agent /root_mount/" in setup
    assert "mv /root/uv /root_mount/" in setup
    assert "mkdir -p /root_mount /trajectories_mount" in setup
    assert [command for command in runs if "cp -r /app /testbed" in command]
    assert runs.index(next(c for c in runs if "run-agent" in c)) > runs.index(setup)
    assert ("delete_egress", tuple(opensandbox.SETUP_ALLOWED_DOMAINS)) not in events
    assert events[-2:] == [("kill",), ("close",)]
    staged = tmp_path / "staged" / "trajectories" / "task.traj.json"
    assert json.loads(staged.read_text()) == {"info": {}}


def test_opensandbox_block_network_allows_proxy_and_closes_setup_domains(monkeypatch, tmp_path):
    events = []
    executor, created = _fake_executor(
        monkeypatch, events, {"./trajectories/task.traj.json": b"{}"}, block_network=True
    )

    _run_agent(executor, tmp_path)

    policy = created[0]["network_policy"]
    assert policy.default_action == "deny"
    assert {rule.target for rule in policy.egress} == {*opensandbox.SETUP_ALLOWED_DOMAINS, "10.0.0.5"}
    delete_index = events.index(("delete_egress", tuple(opensandbox.SETUP_ALLOWED_DOMAINS)))
    agent_index = next(index for index, event in enumerate(events) if event[0] == "run" and "run-agent" in event[1])
    assert delete_index < agent_index


def test_opensandbox_kills_sandbox_when_agent_fails(monkeypatch, tmp_path):
    events = []
    executor, _ = _fake_executor(monkeypatch, events, {}, failing_command="run-agent")

    with pytest.raises(RuntimeError, match="exit 1"):
        _run_agent(executor, tmp_path)

    assert events[-2:] == [("kill",), ("close",)]
    assert not any(event[0] == "read" for event in events)


def test_opensandbox_requires_credentials(monkeypatch):
    monkeypatch.delenv("OPENSANDBOX_DOMAIN", raising=False)
    monkeypatch.delenv("OPENSANDBOX_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENSANDBOX_DOMAIN and OPENSANDBOX_API_KEY"):
        OpenSandboxExecutor()


def test_opensandbox_rejects_unsafe_output_archive(tmp_path):
    with pytest.raises(RuntimeError, match="Unsafe sandbox artifact path"):
        _extract_outputs(_tar_gz({"../escape.txt": b"x"}), tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_get_task_image_prefers_docker_image_and_rejects_sif():
    assert get_task_image({"docker_image": "ghcr.io/a/b:c"}) == "ghcr.io/a/b:c"
    assert get_task_image({"container_formatter": "docker://ghcr.io/a/b:c"}) == "ghcr.io/a/b:c"
    with pytest.raises(RuntimeError, match="OCI docker_image"):
        get_task_image({"instance_id": "x", "container_formatter": "/images/b.sif"})


def test_proxy_binds_to_requested_host(tmp_path):
    async def run():
        async with capture_first_llm_request(
            "http://127.0.0.1:9/v1", tmp_path / "capture.json", host="127.0.0.1"
        ) as proxy_url:
            return proxy_url

    assert asyncio.run(run()).startswith("http://127.0.0.1:")


def _atlas_task(tmp_path, executor, agent_framework=SupportedAgentFrameworks.mini_swe_agent, max_retries=2):
    task = object.__new__(SweAtlasQnAGenerationTask)
    task.cfg = SimpleNamespace(
        agent_framework=agent_framework,
        agent_config="eval/swe-atlas-qna/swe-agent/default",
        max_retries=max_retries,
        min_retry_interval=0,
        max_retry_interval=0,
        setup_timeout=60,
        opensandbox_proxy_host="10.0.0.5",
    )
    task.opensandbox_executor = executor
    task.opensandbox_setup_command = "install-agent"
    task.output_dir = tmp_path / "output"
    task.output_dir.mkdir()
    task.scratch_dir = tmp_path / "scratch"
    task.scratch_dir.mkdir()
    return task


def test_atlas_opensandbox_rollout_copies_outputs_and_prepares_repo(tmp_path):
    calls = []

    class FakeExecutor:
        async def run_agent(self, **kwargs):
            calls.append(kwargs)
            trajectory = kwargs["outputs_destination"] / "trajectories" / "task-1.traj.json"
            trajectory.parent.mkdir(parents=True)
            trajectory.write_text("{}", encoding="utf-8")

    task = _atlas_task(tmp_path, FakeExecutor())
    expected = str(task.output_dir / "trajectories" / "task-1.traj.json")
    data_point = {
        "instance_id": "task-1",
        "docker_image": "ghcr.io/scaleapi/swe-atlas:task",
        "container_formatter": "/images/task.sif",
        "container_repo_dir": "/app",
    }

    result = asyncio.run(task._execute_container_command(data_point, "run-agent", expected, mode="agent", timeout=30))

    assert result == expected
    call = calls[0]
    assert call["image"] == "ghcr.io/scaleapi/swe-atlas:task"
    assert call["setup_command"] == "install-agent"
    assert call["runtime_dirs"] == ("mini-swe-agent", "uv")
    assert call["repo_setup_command"] == "cp -r /app /testbed"
    assert call["proxy_host"] == "10.0.0.5"
    assert call["timeout"] == 30
    for source, target in call["uploads"]:
        assert Path(source).is_file()
        assert target == f"{NEMO_RUN_EVAL_DIR}/{Path(source).name}"
    assert list(task.scratch_dir.iterdir()) == []


def test_atlas_opensandbox_swe_agent_uploads_its_config(tmp_path):
    task = _atlas_task(tmp_path, executor=None, agent_framework=SupportedAgentFrameworks.swe_agent)

    targets = {target for _, target in task._get_opensandbox_uploads()}

    assert any(target.endswith("swe-atlas-qna/swe-agent/default.yaml") for target in targets)


def test_atlas_opensandbox_retries_failures_but_not_timeouts(tmp_path):
    attempts = []

    class FlakyExecutor:
        async def run_agent(self, **kwargs):
            attempts.append(kwargs["log_file"].name)
            if len(attempts) == 1:
                raise RuntimeError("transient")
            trajectory = kwargs["outputs_destination"] / "out.json"
            trajectory.write_text("{}", encoding="utf-8")

    task = _atlas_task(tmp_path, FlakyExecutor())
    data_point = {"instance_id": "task-1", "docker_image": "ghcr.io/a/b:c"}
    expected = str(task.output_dir / "out.json")
    assert asyncio.run(task._execute_container_command(data_point, "cmd", expected, mode="agent")) == expected
    assert attempts == ["task-1_agent_attempt1.log", "task-1_agent_attempt2.log"]

    timeouts = []

    class TimeoutExecutor:
        async def run_agent(self, **kwargs):
            timeouts.append(1)
            raise OpenSandboxCommandTimeout("timed out")

    (tmp_path / "second").mkdir()
    task = _atlas_task(tmp_path / "second", TimeoutExecutor())
    with pytest.raises(ValueError, match="Command timed out"):
        asyncio.run(task._execute_container_command(data_point, "cmd", expected, mode="agent"))
    assert timeouts == [1]


def test_atlas_opensandbox_defers_setup_and_binds_proxy_to_coordinator(tmp_path):
    task = _atlas_task(tmp_path, executor=object())

    task._run_setup(["install uv", "install agent"])

    assert task.opensandbox_setup_command == "install uv && install agent"
    assert task._get_llm_proxy_host() == "10.0.0.5"


def test_atlas_opensandbox_rejects_loopback_proxy_host(monkeypatch):
    monkeypatch.setenv("OPENSANDBOX_DOMAIN", "sandbox.example.com")
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "secret")
    monkeypatch.setattr(opensandbox, "_load_sdk", lambda: (None, None, None, None, None))
    cfg = SweAtlasQnAGenerationConfig(
        input_file="input.jsonl",
        output_file="output.jsonl",
        agent_framework=SupportedAgentFrameworks.mini_swe_agent,
        execution_backend="opensandbox",
        opensandbox_proxy_host="127.0.0.1",
    )

    with pytest.raises(ValueError, match="opensandbox_proxy_host"):
        SweAtlasQnAGenerationTask(cfg)


def test_opensandbox_is_the_default_backend():
    cfg = SweAtlasQnAGenerationConfig(
        input_file="input.jsonl",
        output_file="output.jsonl",
        agent_framework=SupportedAgentFrameworks.mini_swe_agent,
    )

    assert cfg.execution_backend == "opensandbox"
    assert not _uses_apptainer("")
    assert not _uses_apptainer("++agent_framework=openhands ++execution_backend=opensandbox")
    assert _uses_apptainer("++execution_backend=apptainer")
    assert _uses_apptainer("execution_backend=apptainer")
