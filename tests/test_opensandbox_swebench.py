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

import asyncio
import io
import json
import tarfile
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import OmegaConf

from nemo_skills.inference.eval.opensandbox import OpenSandboxExecutor, _extract_outputs
from nemo_skills.inference.eval.swebench import SweBenchGenerationConfig, SweBenchGenerationTask


def _archive(name="sample.traj.json", content=b'{"info":{"submission":"patch"}}'):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo(name)
        member.size = len(content)
        archive.addfile(member, io.BytesIO(content))
    return buffer.getvalue()


@pytest.fixture
def executor(monkeypatch):
    monkeypatch.setenv("OPENSANDBOX_DOMAIN", "https://sandbox.example:443")
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "test-only-key")
    monkeypatch.delenv("OPENSANDBOX_TLS_VERIFY", raising=False)
    return OpenSandboxExecutor(poll_interval_s=0.001)


def _sandbox(executor, monkeypatch, output=b"{}"):
    sandbox = SimpleNamespace(
        commands=SimpleNamespace(
            run=AsyncMock(return_value=SimpleNamespace(id="command-1", error=None)),
            get_command_status=AsyncMock(return_value=SimpleNamespace(running=False, exit_code=0, error=None)),
            get_background_command_logs=AsyncMock(return_value=SimpleNamespace(content="command log\n", cursor=1)),
        ),
        files=SimpleNamespace(write_file=AsyncMock(), read_bytes=AsyncMock(return_value=output)),
        kill=AsyncMock(),
        close=AsyncMock(),
    )
    create = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(executor.Sandbox, "create", create)
    return sandbox, create


def _arguments(tmp_path, mode="agent"):
    output_dir = tmp_path / "results"
    config = tmp_path / "agent.yaml"
    config.write_text("agent-config")
    return dict(
        data_point={
            "instance_id": "owner__repo-1",
            "container_formatter": "docker://swebench/sweb.eval.x86_64.{instance_id}",
            "problem_statement": "Fix café",
            "patch": "gold patch",
            "test_patch": "hidden test patch",
        },
        command="run-agent" if mode == "agent" else "run-verifier",
        expected_file=output_dir / "trajectories/sample.traj.json",
        output_dir=output_dir,
        input_file=tmp_path / "dataset.jsonl",
        setup_command="install-runtime",
        setup_timeout=1200,
        mode=mode,
        agent_framework="mini_swe_agent",
        timeout=60,
        extra_files=[(config, "/agent.yaml", True)],
        proxy_host="10.0.0.2",
        block_network=False,
    )


def test_sdk_connection_reads_env_without_injecting_keys_into_worker(executor):
    connection = executor.ConnectionConfig(**executor.connection_options)
    assert connection.domain == "sandbox.example:443"
    assert connection.protocol == "https"
    assert connection.api_key == "test-only-key"
    assert connection.use_server_proxy is True


def test_missing_credentials_fail_before_creating_sandbox(monkeypatch):
    monkeypatch.delenv("OPENSANDBOX_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENSANDBOX_API_KEY"):
        OpenSandboxExecutor(domain="sandbox.example")


def test_remote_agent_downloads_artifact_and_never_uploads_gold_data(executor, monkeypatch, tmp_path):
    sandbox, create = _sandbox(executor, monkeypatch, _archive())
    args = _arguments(tmp_path)
    result = asyncio.run(executor.execute(**args))
    assert result == str(args["expected_file"])
    assert json.loads(Path(result).read_text())["info"]["submission"] == "patch"
    assert create.call_args.kwargs["image"] == "swebench/sweb.eval.x86_64.owner_1776_repo-1"
    assert "env" not in create.call_args.kwargs
    uploads = sandbox.files.write_file.call_args_list
    assert len(uploads) == 1
    assert uploads[0].args == ("/agent.yaml", b"agent-config")
    commands = "\n".join(call.args[0] for call in sandbox.commands.run.call_args_list)
    assert "mini-swe-agent" in commands
    assert "SWE-bench" not in commands
    assert "apptainer" not in commands
    sandbox.kill.assert_awaited_once()
    sandbox.close.assert_awaited_once()


@pytest.mark.parametrize("framework", ["mini_swe_agent", "swe_agent", "openhands", "opencode", "claude_code"])
def test_verifier_receives_only_its_record_in_a_fresh_sandbox(executor, monkeypatch, tmp_path, framework):
    sandbox, create = _sandbox(executor, monkeypatch, _archive())
    args = _arguments(tmp_path, mode="eval")
    args["agent_framework"] = framework
    asyncio.run(executor.execute(**args))
    record_upload = sandbox.files.write_file.call_args_list[-1]
    assert record_upload.args[0] == "/input_mount/dataset.jsonl"
    assert json.loads(record_upload.args[1]) == args["data_point"]
    assert create.call_args.kwargs["metadata"]["mode"] == "eval"
    assert "SWE-bench" in sandbox.commands.run.call_args_list[0].args[0]
    sandbox.kill.assert_awaited_once()


def test_failed_mutating_command_is_not_replayed_and_worker_is_cleaned_up(executor, monkeypatch, tmp_path):
    sandbox, _ = _sandbox(executor, monkeypatch)
    sandbox.commands.get_command_status.return_value.exit_code = 7
    with pytest.raises(RuntimeError, match="exit 7"):
        asyncio.run(executor.execute(**_arguments(tmp_path)))
    sandbox.commands.run.assert_awaited_once()
    sandbox.kill.assert_awaited_once()
    sandbox.close.assert_awaited_once()
    assert "command log" in (tmp_path / "results/opensandbox_logs/owner__repo-1_agent.log").read_text()


def test_cancelled_execution_terminates_worker(executor, monkeypatch, tmp_path):
    sandbox, _ = _sandbox(executor, monkeypatch)
    sandbox.commands.get_command_status.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(executor.execute(**_arguments(tmp_path)))
    sandbox.kill.assert_awaited_once()
    sandbox.close.assert_awaited_once()


def test_timeout_terminates_worker(executor, monkeypatch, tmp_path):
    sandbox, _ = _sandbox(executor, monkeypatch)
    sandbox.commands.get_command_status.side_effect = asyncio.TimeoutError
    with pytest.raises(RuntimeError, match="timed out"):
        asyncio.run(executor.execute(**_arguments(tmp_path)))
    sandbox.kill.assert_awaited_once()
    sandbox.close.assert_awaited_once()


def test_cleanup_closes_sdk_even_when_termination_fails(executor, monkeypatch, tmp_path):
    sandbox, _ = _sandbox(executor, monkeypatch, _archive())
    sandbox.kill.side_effect = RuntimeError("termination failed")
    with pytest.raises(RuntimeError, match="termination failed"):
        asyncio.run(executor.execute(**_arguments(tmp_path)))
    sandbox.close.assert_awaited_once()


def test_sif_image_is_rejected_before_allocating_worker(executor, monkeypatch, tmp_path):
    _, create = _sandbox(executor, monkeypatch)
    args = _arguments(tmp_path)
    args["data_point"]["container_formatter"] = "/images/{instance_id}.sif"
    with pytest.raises(RuntimeError, match="OCI image"):
        asyncio.run(executor.execute(**args))
    create.assert_not_called()


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape", "folder/../../escape"])
def test_remote_artifacts_cannot_escape_output_directory(tmp_path, name):
    with pytest.raises(RuntimeError, match="Unsafe sandbox artifact"):
        _extract_outputs(_archive(name), tmp_path / "outputs")
    assert not (tmp_path / "escape").exists()


def test_remote_artifact_symlinks_are_rejected(tmp_path):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo("link")
        member.type = tarfile.SYMTYPE
        member.linkname = "/tmp/escape"
        archive.addfile(member)
    with pytest.raises(RuntimeError, match="Unsupported sandbox artifact"):
        _extract_outputs(buffer.getvalue(), tmp_path)


@pytest.mark.parametrize("proxy_host", [None, "10.0.0.2"])
@pytest.mark.parametrize("framework", ["mini_swe_agent", "swe_agent", "openhands", "opencode", "claude_code"])
def test_opensandbox_setup_stays_remote_and_separates_agent_from_verifier(
    executor, monkeypatch, tmp_path, proxy_host, framework
):
    detect = MagicMock(return_value="10.0.0.3")
    monkeypatch.setattr(executor, "detect_proxy_host", detect)
    monkeypatch.setattr(SweBenchGenerationTask, "_execute_local_command", AsyncMock())
    monkeypatch.setattr("nemo_skills.inference.eval.opensandbox.OpenSandboxExecutor", lambda **kwargs: executor)
    monkeypatch.setattr("nemo_skills.inference.eval.swebench.SCRATCH_DIR", tmp_path)
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n")
    cfg = SweBenchGenerationConfig(
        _init_nested=True,
        input_file=str(dataset),
        output_file=str(tmp_path / "output.jsonl"),
        agent_framework=framework,
        multilingual=True,
        execution_backend="opensandbox",
        opensandbox_proxy_host=proxy_host,
        server=OmegaConf.create({"base_url": "http://gpu.example:8000/v1"}),
    )
    task = SweBenchGenerationTask(cfg)
    task._execute_local_command.assert_not_called()
    package = {
        "mini_swe_agent": "mini-swe-agent",
        "swe_agent": "SWE-agent",
        "openhands": "OpenHands",
        "opencode": "opencode-ai",
        "claude_code": "@anthropic-ai/claude-code",
    }[framework]
    assert package in task.opensandbox_setup_commands["agent"]
    assert "SWE-bench" not in task.opensandbox_setup_commands["agent"]
    assert "SWE-bench" in task.opensandbox_setup_commands["eval"]
    assert package not in task.opensandbox_setup_commands["eval"]
    assert task.api_base == "http://gpu.example:8000/v1"
    assert task.cfg.opensandbox_proxy_host == (proxy_host or "10.0.0.3")
    assert detect.call_count == (1 if proxy_host is None else 0)


@pytest.mark.parametrize("instance_id", ["owner__repo-1", "axios__axios-4731"])
def test_backend_dispatch_preserves_native_artifact_contract(executor, monkeypatch, tmp_path, instance_id):
    sandbox, _ = _sandbox(executor, monkeypatch, _archive())
    args = _arguments(tmp_path)
    args["data_point"]["instance_id"] = instance_id
    args["mode"] = "eval"
    task = object.__new__(SweBenchGenerationTask)
    task.cfg = SimpleNamespace(
        input_file=str(args["input_file"]),
        setup_timeout=args["setup_timeout"],
        agent_framework="mini_swe_agent",
        opensandbox_proxy_host=args["proxy_host"],
        block_network=False,
    )
    task.opensandbox_executor = executor
    task.opensandbox_setup_commands = {"eval": "install-runtime"}
    task.output_dir = args["output_dir"]
    result = asyncio.run(
        task._execute_container_command(
            args["data_point"],
            args["command"],
            str(args["expected_file"]),
            "eval",
            timeout=60,
            extra_mounts=args["extra_files"],
        )
    )
    assert json.loads(Path(result).read_text())["info"]["submission"] == "patch"
    sandbox.kill.assert_awaited_once()


def test_remote_infrastructure_failure_is_not_scored_as_an_incorrect_patch(tmp_path):
    prediction = tmp_path / "prediction.jsonl"
    prediction.write_text(json.dumps({"instance_id": "owner__repo-1", "model_patch": "patch"}))
    task = object.__new__(SweBenchGenerationTask)
    task.output_dir = tmp_path
    task.cfg = SimpleNamespace(
        evaluate=True,
        dataset_type="swe_bench",
        swebench_tests_timeout=60,
        input_file=str(tmp_path / "dataset.jsonl"),
    )
    task._run_agent = AsyncMock(return_value=str(prediction))
    task._execute_container_command = AsyncMock(side_effect=RuntimeError("sandbox service unavailable"))

    async def run():
        task.rollout_semaphore = asyncio.Semaphore(1)
        task.eval_semaphore = asyncio.Semaphore(1)
        return await task.process_single_datapoint({"instance_id": "owner__repo-1"}, [])

    with pytest.raises(RuntimeError, match="sandbox service unavailable"):
        asyncio.run(run())


def test_hydra_numeric_resource_values_are_normalized_for_sdk(monkeypatch):
    monkeypatch.setenv("OPENSANDBOX_DOMAIN", "sandbox.example")
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "test-only-key")
    executor = OpenSandboxExecutor(resources={"cpu": 4, "memory": "16Gi"})
    assert executor.resources == {"cpu": "4", "memory": "16Gi"}


def test_proxy_address_uses_route_to_service_without_sending_packets(executor, monkeypatch):
    import socket

    resolve = MagicMock(return_value=[(socket.AF_INET, socket.SOCK_DGRAM, 17, "", ("10.50.0.1", 443))])
    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    route = MagicMock()
    route.getsockname.return_value = ("10.60.0.2", 54321)
    factory = MagicMock()
    factory.return_value.__enter__.return_value = route
    monkeypatch.setattr(socket, "socket", factory)
    assert executor.detect_proxy_host() == "10.60.0.2"
    resolve.assert_called_once_with("sandbox.example", 443, socket.AF_INET, socket.SOCK_DGRAM)
    route.connect.assert_called_once_with(("10.50.0.1", 443))
    route.send.assert_not_called()
    route.sendto.assert_not_called()
    factory.return_value.__exit__.assert_called_once()


def test_proxy_address_falls_back_to_allocated_node_hostname(executor, monkeypatch):
    import socket

    monkeypatch.setattr(socket, "getaddrinfo", MagicMock(side_effect=OSError("no route")))
    monkeypatch.setattr(socket, "gethostname", lambda: "allocated-dfw-node")
    resolve = MagicMock(return_value="10.60.0.3")
    monkeypatch.setattr(socket, "gethostbyname", resolve)
    assert executor.detect_proxy_host() == "10.60.0.3"
    resolve.assert_called_once_with("allocated-dfw-node")


@pytest.mark.parametrize("address", ["127.0.0.2", "0.0.0.0", "169.254.1.2"])
def test_proxy_address_rejects_unreachable_local_addresses(executor, monkeypatch, address):
    import socket

    monkeypatch.setattr(socket, "getaddrinfo", MagicMock(side_effect=OSError("no route")))
    monkeypatch.setattr(socket, "gethostbyname", lambda hostname: address)
    with pytest.raises(RuntimeError, match="opensandbox_proxy_host"):
        executor.detect_proxy_host()


@pytest.mark.parametrize(
    "value,expected", [("true", True), ("1", True), ("false", False), ("0", False), (" FALSE ", False)]
)
def test_tls_verification_reads_env(executor, monkeypatch, value, expected):
    monkeypatch.setenv("OPENSANDBOX_TLS_VERIFY", value)
    assert OpenSandboxExecutor().tls_verify is expected


def test_tls_verification_defaults_to_enabled(executor):
    assert executor.tls_verify is True


def test_invalid_tls_verification_setting_fails_explicitly(executor, monkeypatch):
    monkeypatch.setenv("OPENSANDBOX_TLS_VERIFY", "invalid")
    with pytest.raises(ValueError, match="OPENSANDBOX_TLS_VERIFY"):
        OpenSandboxExecutor()


@pytest.mark.parametrize("create_fails", [False, True])
def test_tls_transport_disables_verification_and_closes_on_success_or_create_failure(
    executor, monkeypatch, tmp_path, create_fails
):
    import httpx

    executor.tls_verify = False
    sandbox, create = _sandbox(executor, monkeypatch, _archive())
    if create_fails:
        create.side_effect = RuntimeError("create failed")
    factory = httpx.AsyncHTTPTransport
    transports = []

    def track_transport(**kwargs):
        assert kwargs["verify"] is False
        transport = factory(**kwargs)
        transport._pool.aclose = AsyncMock(wraps=transport._pool.aclose)
        transports.append(transport)
        return transport

    monkeypatch.setattr(httpx, "AsyncHTTPTransport", track_transport)
    if create_fails:
        with pytest.raises(RuntimeError, match="create failed"):
            asyncio.run(executor.execute(**_arguments(tmp_path)))
    else:
        asyncio.run(executor.execute(**_arguments(tmp_path)))
        sandbox.close.assert_awaited_once()
    assert len(transports) == 1
    transports[0]._pool.aclose.assert_awaited_once()
    connection = create.call_args.kwargs["connection_config"]
    assert connection.transport is not None
    assert connection._owns_transport is False


@pytest.mark.parametrize("framework", ["mini_swe_agent", "swe_agent", "openhands", "opencode", "claude_code"])
@pytest.mark.parametrize("multilingual", [False, True])
def test_native_harness_runs_through_remote_backend_and_returns_prediction(
    executor, monkeypatch, tmp_path, framework, multilingual
):
    args = _arguments(tmp_path)
    data_point = {
        **args["data_point"],
        "base_commit": "abc123",
        "language": "python",
        "dataset_name": "SWE-bench/SWE-bench_Multilingual",
        "repo": "owner/repo",
    }
    instance_id = data_point["instance_id"]
    patch = "diff --git a/main.py b/main.py\n"
    if framework == "mini_swe_agent":
        files = {f"{instance_id}.traj.json": json.dumps({"info": {"submission": patch}})}
    elif framework == "swe_agent":
        files = {f"{instance_id}.pred": json.dumps({"instance_id": instance_id, "model_patch": patch})}
    elif framework == "openhands":
        files = {
            "output.jsonl": json.dumps(
                {
                    "instance_id": instance_id,
                    "test_result": {"git_patch": patch},
                    "metadata": {"llm_config": {"model": "served-model"}},
                }
            )
        }
    else:
        files = {"model.patch": patch}
        if framework == "claude_code":
            files["claude-code.exit-code"] = "0\n"
    files["worker.log"] = "remote artifact"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in files.items():
            content = content.encode()
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    sandbox, create = _sandbox(executor, monkeypatch, buffer.getvalue())
    monkeypatch.setattr(SweBenchGenerationTask, "_execute_local_command", AsyncMock())
    monkeypatch.setattr("nemo_skills.inference.eval.opensandbox.OpenSandboxExecutor", lambda **kwargs: executor)
    monkeypatch.setattr("nemo_skills.inference.eval.swebench.SCRATCH_DIR", tmp_path)
    proxy_calls = []

    @asynccontextmanager
    async def capture(upstream, output_path, **kwargs):
        assert upstream == "http://gpu.example:8000/v1"
        proxy_calls.append(kwargs)
        yield "http://10.0.0.2:45678/v1"

    monkeypatch.setattr("nemo_skills.inference.eval.swebench.capture_first_llm_request", capture)
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps(data_point) + "\n")
    cfg = SweBenchGenerationConfig(
        _init_nested=True,
        input_file=str(dataset),
        output_file=str(tmp_path / "results" / "output.jsonl"),
        agent_framework=framework,
        multilingual=multilingual,
        execution_backend="opensandbox",
        opensandbox_proxy_host="10.0.0.2",
        server=OmegaConf.create({"base_url": "http://gpu.example:8000/v1", "model": "served-model"}),
    )
    cfg.inference.extra_body = OmegaConf.create({"chat_template_kwargs": {"enable_thinking": False}})
    task = SweBenchGenerationTask(cfg)
    prediction = Path(asyncio.run(task._run_agent(data_point)))
    assert json.loads(prediction.read_text())["model_patch"] == patch
    assert (prediction.parent / "worker.log").read_text() == "remote artifact"
    task._execute_local_command.assert_not_called()
    assert len(proxy_calls) == 1
    assert proxy_calls[0]["host"] == "10.0.0.2"
    if framework in {"mini_swe_agent", "swe_agent", "claude_code"}:
        assert proxy_calls[0]["request_transform"] is not None
    commands = "\n".join(call.args[0] for call in sandbox.commands.run.call_args_list)
    uploads = "\n".join(call.args[1].decode() for call in sandbox.files.write_file.call_args_list)
    assert "http://10.0.0.2:45678" in commands + uploads
    paths = {
        "mini_swe_agent": ["mini-swe-agent", "uv"],
        "swe_agent": ["SWE-agent", "uv"],
        "openhands": ["OpenHands", "uv", "tmux", "jq"],
        "opencode": ["node"],
        "claude_code": ["node"],
    }[framework]
    for path in paths:
        assert f"cp -a /root/{path} /root_mount/" in commands
    assert "cp -a /root/SWE-bench /root_mount/" not in commands
    if framework in {"opencode", "claude_code"}:
        assert "cp -a /root/uv /root_mount/" not in commands
    for upload in sandbox.files.write_file.call_args_list:
        content = upload.args[1].decode()
        assert "gold patch" not in content
        assert "hidden test patch" not in content
    if framework == "openhands":
        record = json.loads(sandbox.files.write_file.call_args_list[0].args[1])
        assert record["instance_id"] == instance_id
        assert "patch" not in record and "test_patch" not in record
    create.assert_awaited_once()
    sandbox.kill.assert_awaited_once()
    sandbox.close.assert_awaited_once()


def test_sdk_health_check_authenticates_through_server_proxy(executor):
    import httpx
    from opensandbox.adapters.health_adapter import HealthAdapter

    requests = []

    def handler(request):
        requests.append(request)
        if request.headers.get("OPEN-SANDBOX-API-KEY") != "test-only-key":
            return httpx.Response(401, json={"code": "MISSING_API_KEY"})
        return httpx.Response(200)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            config = executor.ConnectionConfig(**executor.connection_options, transport=client._transport)
            endpoint = SimpleNamespace(endpoint="sandbox.example/v1/sandboxes/id/proxy/44772", headers={})
            health = HealthAdapter(config, endpoint)
            try:
                assert await health.ping("id") is True
            finally:
                await health._httpx_client.aclose()

    asyncio.run(run())
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/sandboxes/id/proxy/44772/ping"
    assert requests[0].headers["OPEN-SANDBOX-API-KEY"] == "test-only-key"
