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
import hashlib
import io
import json
import shlex
import subprocess
import tarfile
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import OmegaConf

from nemo_skills.inference.eval.git_sanitization import _LEGACY_GIT_CLEANUP
from nemo_skills.inference.eval.opensandbox import (
    OpenSandboxExecutor,
    _archive_input_directory,
    _extract_outputs,
    _is_transient_create_error,
    _resolve_dockerhub_image_digest,
)
from nemo_skills.inference.eval.swebench import (
    SupportedDatasetTypes,
    SweBenchGenerationConfig,
    SweBenchGenerationTask,
)
from nemo_skills.inference.eval.swebench_pro_v2 import SweBenchProV2GenerationTask


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
        delete_egress_rules=AsyncMock(),
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
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n")
    return dict(
        data_point={
            "instance_id": "owner__repo-1",
            "docker_image": "swebench/sweb.eval.x86_64.owner_1776_repo-1",
            "problem_statement": "Fix café",
            "patch": "gold patch",
            "test_patch": "hidden test patch",
        },
        command="run-agent" if mode == "agent" else "run-verifier",
        expected_file=output_dir / "trajectories/sample.traj.json",
        output_dir=output_dir,
        input_file=dataset,
        setup_command="install-runtime",
        setup_timeout=1200,
        mode=mode,
        agent_framework="mini_swe_agent",
        timeout=60,
        extra_files=[(config, "/agent.yaml", True)],
        proxy_host="10.0.0.2",
        block_network=False,
    )


def _configure_task_scratch(task, tmp_path):
    task.scratch_dir = tmp_path / "task-scratch"
    task.input_dir = task.scratch_dir / "input"
    task._scratch_artifacts = {}


def _sandbox_error(name, *, status_code=None):
    bases = (Exception,)
    if name != "SandboxApiException":
        bases = (type("SandboxApiException", (Exception,), {}),)
    error = type(name, bases, {})()
    error.status_code = status_code
    return error


@pytest.mark.parametrize(
    "image,repository,tag",
    [
        (
            "jefzda/sweap-images:nodebb.nodebb-NodeBB__NodeBB-abc",
            "jefzda/sweap-images",
            "nodebb.nodebb-NodeBB__NodeBB-abc",
        ),
        ("docker.io/jefzda/sweap-images:NodeBB", "jefzda/sweap-images", "NodeBB"),
        ("ubuntu", "library/ubuntu", "latest"),
    ],
)
def test_dockerhub_digest_resolution_preserves_tag_case(monkeypatch, image, repository, tag):
    import httpx

    manifest = b'{"schemaVersion":2,"layers":[]}'
    digest = "sha256:" + hashlib.sha256(manifest).hexdigest()
    requests = []

    def respond(request):
        requests.append(request)
        if request.url.host == "auth.docker.io":
            assert request.url.params["scope"] == f"repository:{repository}:pull"
            assert "authorization" not in request.headers
            return httpx.Response(200, json={"token": "anonymous-registry-token"})
        assert request.url.host == "registry-1.docker.io"
        assert request.method == "HEAD"
        assert request.url.path == f"/v2/{repository}/manifests/{tag}"
        assert request.headers["Authorization"] == "Bearer anonymous-registry-token"
        assert "OPEN-SANDBOX-API-KEY" not in request.headers
        return httpx.Response(200, headers={"Docker-Content-Digest": digest})

    client_factory = httpx.AsyncClient
    transport = httpx.MockTransport(respond)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: client_factory(transport=transport, **kwargs))
    assert asyncio.run(_resolve_dockerhub_image_digest(image)) == f"{repository}@{digest}"
    assert len(requests) == 2


@pytest.mark.parametrize("failure", ["authentication", "missing_manifest", "missing_digest", "invalid_digest"])
def test_dockerhub_resolution_fails_without_using_an_unverified_digest(monkeypatch, failure):
    import httpx

    def respond(request):
        if request.url.host == "auth.docker.io":
            return httpx.Response(401 if failure == "authentication" else 200, json={"token": "anonymous"})
        if failure == "missing_manifest":
            return httpx.Response(404, json={"errors": [{"code": "MANIFEST_UNKNOWN"}]})
        headers = {} if failure == "missing_digest" else {"Docker-Content-Digest": "sha256:not-a-digest"}
        return httpx.Response(200, headers=headers)

    client_factory = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kwargs: client_factory(transport=httpx.MockTransport(respond), **kwargs),
    )
    expected_error = ValueError if failure in {"missing_digest", "invalid_digest"} else httpx.HTTPStatusError
    with pytest.raises(expected_error):
        asyncio.run(_resolve_dockerhub_image_digest("jefzda/sweap-images:NodeBB"))


def test_digest_resolution_rejects_other_registry_tags():
    with pytest.raises(ValueError, match="public Docker Hub"):
        asyncio.run(_resolve_dockerhub_image_digest("private.example:5000/team/image:Tag"))


def test_digest_resolution_is_opt_in_and_preserves_existing_digests(executor, monkeypatch):
    resolve = AsyncMock(side_effect=AssertionError("must not resolve this reference"))
    monkeypatch.setattr("nemo_skills.inference.eval.opensandbox._resolve_dockerhub_image_digest", resolve)
    image = "jefzda/sweap-images:NodeBB"
    assert asyncio.run(executor._resolve_image(image)) == image
    executor.resolve_image_tags_to_digests = True
    pinned = "private.example/team/image@sha256:" + "1" * 64
    assert asyncio.run(executor._resolve_image(pinned)) == pinned
    assert asyncio.run(_resolve_dockerhub_image_digest(pinned)) == pinned
    resolve.assert_not_awaited()


def test_digest_resolution_caches_concurrent_requests(executor, monkeypatch):
    executor = OpenSandboxExecutor(resolve_image_tags_to_digests=True)
    image = "jefzda/sweap-images:NodeBB"
    pinned = "jefzda/sweap-images@sha256:" + "1" * 64
    resolve = AsyncMock(return_value=pinned)
    monkeypatch.setattr("nemo_skills.inference.eval.opensandbox._resolve_dockerhub_image_digest", resolve)

    async def run():
        assert await asyncio.gather(executor._resolve_image(image), executor._resolve_image(image)) == [pinned, pinned]
        assert await executor._resolve_image(image) == pinned

    asyncio.run(run())
    resolve.assert_awaited_once_with(image)


def test_agent_and_verifier_use_the_resolved_digest(executor, monkeypatch, tmp_path):
    executor = OpenSandboxExecutor(resolve_image_tags_to_digests=True, poll_interval_s=0.001)
    sandbox, create = _sandbox(executor, monkeypatch, _archive())
    args = _arguments(tmp_path)
    image = "jefzda/sweap-images:NodeBB"
    pinned = "jefzda/sweap-images@sha256:" + "1" * 64
    args["data_point"]["docker_image"] = image
    resolve = AsyncMock(return_value=pinned)
    monkeypatch.setattr("nemo_skills.inference.eval.opensandbox._resolve_dockerhub_image_digest", resolve)

    asyncio.run(executor.execute(**args))
    args["mode"] = "eval"
    asyncio.run(executor.execute(**args))

    assert [call.kwargs["image"] for call in create.call_args_list] == [pinned, pinned]
    resolve.assert_awaited_once_with(image)
    assert sandbox.kill.await_count == 2


def test_resolution_failure_does_not_create_a_sandbox(executor, monkeypatch, tmp_path):
    executor = OpenSandboxExecutor(resolve_image_tags_to_digests=True)
    _, create = _sandbox(executor, monkeypatch)
    resolve = AsyncMock(side_effect=ValueError("registry resolution failed"))
    monkeypatch.setattr("nemo_skills.inference.eval.opensandbox._resolve_dockerhub_image_digest", resolve)
    with pytest.raises(ValueError, match="registry resolution failed"):
        asyncio.run(executor.execute(**_arguments(tmp_path)))
    create.assert_not_called()


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


@pytest.mark.parametrize(
    "error",
    [
        _sandbox_error("SandboxReadyTimeoutException"),
        _sandbox_error("SandboxTimeoutException"),
        _sandbox_error("SandboxConnectionException"),
        _sandbox_error("SandboxApiException", status_code=None),
        _sandbox_error("SandboxApiException", status_code=408),
        _sandbox_error("SandboxApiException", status_code=429),
        _sandbox_error("SandboxApiException", status_code=503),
    ],
)
def test_transient_sandbox_creation_errors_are_retryable(error):
    assert _is_transient_create_error(error) is True


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
def test_client_sandbox_creation_errors_are_not_retryable(status_code):
    assert _is_transient_create_error(_sandbox_error("SandboxApiException", status_code=status_code)) is False


def test_transient_sandbox_creation_failure_retries_then_succeeds(executor, monkeypatch, tmp_path):
    sandbox, create = _sandbox(executor, monkeypatch, _archive())
    create.side_effect = [_sandbox_error("SandboxReadyTimeoutException"), sandbox]
    executor.create_retry_min_s = executor.create_retry_max_s = 0

    asyncio.run(executor.execute(**_arguments(tmp_path)))

    assert create.await_count == 2


def test_sandbox_creation_retry_exhaustion_reraises_last_error(executor, monkeypatch, tmp_path):
    error = _sandbox_error("SandboxConnectionException")
    _, create = _sandbox(executor, monkeypatch)
    create.side_effect = error
    executor.create_retries = 2
    executor.create_retry_min_s = executor.create_retry_max_s = 0

    with pytest.raises(type(error)):
        asyncio.run(executor.execute(**_arguments(tmp_path)))

    assert create.await_count == 3


def test_non_retryable_sandbox_creation_failure_is_not_retried(executor, monkeypatch, tmp_path):
    error = _sandbox_error("SandboxApiException", status_code=401)
    _, create = _sandbox(executor, monkeypatch)
    create.side_effect = error

    with pytest.raises(type(error)):
        asyncio.run(executor.execute(**_arguments(tmp_path)))

    create.assert_awaited_once()


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


@pytest.mark.parametrize(
    "framework",
    ["mini_swe_agent", "swe_agent", "openhands", "opencode", "claude_code", "gold_patch", "model_patch"],
)
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


def test_locked_agent_and_verifier_use_separate_network_policies(executor, monkeypatch, tmp_path):
    agent_sandbox, agent_create = _sandbox(executor, monkeypatch, _archive())
    agent_args = _arguments(tmp_path, mode="agent")
    agent_args["block_network"] = True
    asyncio.run(executor.execute(**agent_args))

    assert agent_create.call_args.kwargs["network_policy"] is not None
    agent_sandbox.delete_egress_rules.assert_awaited_once()

    verifier_sandbox, verifier_create = _sandbox(executor, monkeypatch, _archive())
    verifier_args = _arguments(tmp_path, mode="eval")
    verifier_args["block_network"] = True
    verifier_args["stage_eval_harness"] = False
    asyncio.run(executor.execute(**verifier_args))

    assert verifier_create.call_args.kwargs["network_policy"] is None
    verifier_sandbox.delete_egress_rules.assert_not_awaited()


@pytest.mark.parametrize("framework", ["gold_patch", "model_patch"])
def test_supplied_patch_cannot_run_in_agent_sandbox(executor, monkeypatch, tmp_path, framework):
    _, create = _sandbox(executor, monkeypatch)
    args = _arguments(tmp_path, mode="agent")
    args["agent_framework"] = framework

    with pytest.raises(RuntimeError, match="does not run an agent sandbox"):
        asyncio.run(executor.execute(**args))

    create.assert_not_called()


@pytest.mark.parametrize("repo_setup_timeout_s", [None, 600])
@pytest.mark.parametrize("mode", ["agent", "eval"])
def test_repository_preparation_timeout_and_sandbox_lifetime(
    executor, monkeypatch, tmp_path, repo_setup_timeout_s, mode
):
    if repo_setup_timeout_s is not None:
        executor = OpenSandboxExecutor(repo_setup_timeout_s=repo_setup_timeout_s)
    sandbox, create = _sandbox(executor, monkeypatch, _archive())
    args = _arguments(tmp_path, mode=mode)
    args["data_point"]["pre_commands"] = "git gc --prune=now --aggressive"
    asyncio.run(executor.execute(**args))
    expected_timeout = 300 if repo_setup_timeout_s is None else repo_setup_timeout_s
    preparation_calls = [call for call in sandbox.commands.run.call_args_list if "git gc" in call.args[0]]
    if mode == "agent":
        assert len(preparation_calls) == 1
        assert preparation_calls[0].kwargs["opts"].timeout.total_seconds() == expected_timeout
    else:
        assert not preparation_calls
    expected_lifetime = executor.ready_timeout_s + args["setup_timeout"] + args["timeout"] + 300
    if mode == "agent":
        expected_lifetime += expected_timeout
    assert create.call_args.kwargs["timeout"].total_seconds() == expected_lifetime


def test_swe_bench_pro_repository_is_staged_from_app(executor, monkeypatch, tmp_path):
    sandbox, _ = _sandbox(executor, monkeypatch, _archive())
    args = _arguments(tmp_path)
    args["data_point"]["container_repo_dir"] = "/app"
    args["data_point"]["pre_commands"] = "git status --short"

    asyncio.run(executor.execute(**args))

    commands = [call.args[0] for call in sandbox.commands.run.call_args_list]
    repo_setup_command = next(command for command in commands if "git status --short" in command)
    assert "cd /app" in repo_setup_command
    assert "cp -r /app /testbed" in repo_setup_command


def test_repository_is_discovered_from_ordered_candidates(executor, monkeypatch, tmp_path):
    sandbox, _ = _sandbox(executor, monkeypatch, _archive())
    args = _arguments(tmp_path)
    args["data_point"]["container_repo_dirs"] = ["/app", "/testbed"]
    args["data_point"]["pre_commands"] = "git status --short"

    asyncio.run(executor.execute(**args))

    commands = [call.args[0] for call in sandbox.commands.run.call_args_list]
    repo_setup_command = next(command for command in commands if "git status --short" in command)
    assert "for candidate in /app /testbed" in repo_setup_command
    assert 'cd "$container_repo_dir"' in repo_setup_command
    assert 'cp -r "$container_repo_dir" /testbed' in repo_setup_command


@pytest.mark.parametrize("repo_setup_timeout_s", [0, -1])
def test_repository_preparation_timeout_must_be_positive(executor, repo_setup_timeout_s):
    with pytest.raises(ValueError, match="must be positive"):
        OpenSandboxExecutor(repo_setup_timeout_s=repo_setup_timeout_s)


@pytest.mark.parametrize("exit_code", [-1, 0])
def test_command_failure_preserves_remote_error(executor, monkeypatch, tmp_path, exit_code):
    sandbox, _ = _sandbox(executor, monkeypatch)
    status = sandbox.commands.get_command_status.return_value
    status.exit_code = exit_code
    status.error = "context deadline exceeded during repository preparation"

    with pytest.raises(RuntimeError, match="context deadline exceeded during repository preparation"):
        asyncio.run(executor.execute(**_arguments(tmp_path)))

    sandbox.commands.run.assert_awaited_once()
    sandbox.kill.assert_awaited_once()
    sandbox.close.assert_awaited_once()


def test_legacy_cleanup_preserves_packed_main_in_opensandbox(executor, monkeypatch, tmp_path):
    repo = tmp_path / "repository"

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    git("-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "--allow-empty", "-m", "base")
    original_head = git("rev-parse", "HEAD")
    git("tag", "future")
    git("branch", "other")
    git("pack-refs", "--all")
    assert not (repo / ".git/refs/heads/main").exists()

    sandbox, _ = _sandbox(executor, monkeypatch, _archive())
    args = _arguments(tmp_path)
    args["data_point"]["pre_commands"] = "git checkout -B main && " + _LEGACY_GIT_CLEANUP
    asyncio.run(executor.execute(**args))

    preparation = next(call.args[0] for call in sandbox.commands.run.call_args_list if "git gc" in call.args[0])
    preparation = shlex.split(preparation)[2].replace("cd /testbed", "cd " + shlex.quote(str(repo)), 1)
    subprocess.run(
        ["bash", "-c", preparation],
        check=True,
        capture_output=True,
        text=True,
    )
    assert git("rev-parse", "HEAD") == original_head
    assert git("for-each-ref", "--format=%(refname)") == "refs/heads/main"


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
    args["data_point"]["docker_image"] = "/images/instance.sif"
    with pytest.raises(RuntimeError, match="OCI image"):
        asyncio.run(executor.execute(**args))
    create.assert_not_called()


def test_missing_docker_image_is_rejected_before_allocating_worker(executor, monkeypatch, tmp_path):
    _, create = _sandbox(executor, monkeypatch)
    args = _arguments(tmp_path)
    del args["data_point"]["docker_image"]
    with pytest.raises(RuntimeError, match="requires a docker_image"):
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


def test_directory_input_archive_preserves_tree_and_modes(tmp_path):
    source = tmp_path / "tests"
    nested = source / "nested"
    nested.mkdir(parents=True)
    script = source / "test.sh"
    script.write_text("#!/bin/bash\n")
    script.chmod(0o755)
    (nested / "config.json").write_text("{}")

    with tarfile.open(fileobj=io.BytesIO(_archive_input_directory(source)), mode="r:gz") as archive:
        assert {member.name for member in archive.getmembers()} == {"nested", "nested/config.json", "test.sh"}
        assert archive.getmember("test.sh").mode & 0o777 == 0o755
        assert archive.extractfile("nested/config.json").read() == b"{}"


def test_directory_input_archive_rejects_symlinks(tmp_path):
    source = tmp_path / "tests"
    source.mkdir()
    (source / "target").write_text("content")
    (source / "link").symlink_to("target")

    with pytest.raises(RuntimeError, match="cannot contain links"):
        _archive_input_directory(source)


def test_harbor_verifier_stages_directory_without_swebench_runtime(executor, monkeypatch, tmp_path):
    sandbox, _ = _sandbox(executor, monkeypatch, _archive())
    args = _arguments(tmp_path, mode="eval")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text("#!/bin/bash\n")
    patch = tmp_path / "model.patch"
    patch.write_text("diff --git a/a b/a\n")
    args["extra_files"] = [
        (tests_dir, "/tests", True),
        (patch, "/patch_mount/model.patch", True),
    ]
    args["stage_eval_harness"] = False

    asyncio.run(executor.execute(**args))

    commands = [call.args[0] for call in sandbox.commands.run.call_args_list]
    assert "SWE-bench" not in commands[0]
    assert any("tar -xzf /tmp/nemo-skills-input-0.tar.gz -C /tests" in command for command in commands)
    uploads = sandbox.files.write_file.call_args_list
    assert uploads[0].args[0] == "/tmp/nemo-skills-input-0.tar.gz"
    assert uploads[1].args == ("/patch_mount/model.patch", patch.read_bytes())
    assert uploads[2].args[0] == "/input_mount/dataset.jsonl"


def test_swe_bench_pro_v2_verifier_returns_remote_reward(executor, monkeypatch, tmp_path):
    instance_id = "instance_owner__repo-abc"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, content in {
            ".eval_done": b"",
            "reward.json": b'{"reward": 1}',
            "verifier.log": b"passed",
        }.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    sandbox, create = _sandbox(executor, monkeypatch, buffer.getvalue())
    tests_dir = tmp_path / "tasks" / instance_id / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test.sh").write_text("#!/bin/bash\n")
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n")

    task = object.__new__(SweBenchProV2GenerationTask)
    task.output_dir = tmp_path / "results"
    task.output_dir.mkdir()
    _configure_task_scratch(task, tmp_path)
    task.opensandbox_executor = executor
    task.opensandbox_setup_commands = {"eval": "install-runtime"}
    task.cfg = SimpleNamespace(
        tasks_dir=str(tmp_path / "tasks"),
        eval_config={},
        use_agent_timeouts=True,
        use_verifier_timeouts=True,
        swebench_tests_timeout=60,
        input_file=str(dataset),
        setup_timeout=1200,
        agent_framework="mini_swe_agent",
        opensandbox_proxy_host="10.0.0.2",
        block_network=True,
        dataset_type=SupportedDatasetTypes.swe_bench_pro_v2,
    )
    data_point = {
        "instance_id": instance_id,
        "docker_image": "ghcr.io/scaleapi/swe-bench_pro-v2:instance_owner__repo-abc",
        "verifier_timeout_sec": 60,
    }

    metrics = asyncio.run(task._run_v2_verifier(data_point, "diff --git a/a b/a\n"))

    assert metrics["resolved"] is True
    assert metrics["reward"] == 1.0
    assert create.call_args.kwargs["metadata"]["mode"] == "eval"
    assert create.call_args.kwargs["network_policy"] is None
    assert (task.output_dir / f"eval-outputs/{instance_id}/verifier.log").read_text() == "passed"
    commands = "\n".join(call.args[0] for call in sandbox.commands.run.call_args_list)
    assert "SWE-bench" not in commands
    assert "bash /tests/test.sh" in commands


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


def test_swe_bench_pro_agent_and_verifier_setup_stays_in_opensandbox(executor, monkeypatch, tmp_path):
    monkeypatch.setattr(executor, "detect_proxy_host", MagicMock(return_value="10.0.0.3"))
    monkeypatch.setattr(SweBenchGenerationTask, "_execute_local_command", AsyncMock())
    monkeypatch.setattr("nemo_skills.inference.eval.opensandbox.OpenSandboxExecutor", lambda **kwargs: executor)
    monkeypatch.setattr("nemo_skills.inference.eval.swebench.SCRATCH_DIR", tmp_path)
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n")
    cfg = SweBenchGenerationConfig(
        _init_nested=True,
        input_file=str(dataset),
        output_file=str(tmp_path / "output.jsonl"),
        agent_framework="mini_swe_agent",
        multilingual=True,
        dataset_type=SupportedDatasetTypes.swe_bench_pro,
        eval_harness_repo="https://github.com/wasiahmad/SWE-bench_Pro-os.git",
        execution_backend="opensandbox",
        server=OmegaConf.create({"base_url": "http://gpu.example:8000/v1"}),
    )

    task = SweBenchGenerationTask(cfg)

    task._execute_local_command.assert_not_called()
    assert "mini-swe-agent" in task.opensandbox_setup_commands["agent"]
    assert "SWE-bench_Pro-os.git" in task.opensandbox_setup_commands["eval"]
    assert "apk add" in task.opensandbox_setup_commands["agent"]
    assert "apk add" in task.opensandbox_setup_commands["eval"]


@pytest.mark.parametrize("framework", ["gold_patch", "model_patch"])
def test_supplied_patch_setup_skips_model_proxy(executor, monkeypatch, tmp_path, framework):
    detect = MagicMock(side_effect=AssertionError("supplied patches do not need a model proxy"))
    monkeypatch.setattr(executor, "detect_proxy_host", detect)
    monkeypatch.setattr(SweBenchGenerationTask, "_execute_local_command", AsyncMock())
    monkeypatch.setattr("nemo_skills.inference.eval.opensandbox.OpenSandboxExecutor", lambda **kwargs: executor)
    monkeypatch.setattr("nemo_skills.inference.eval.swebench.SCRATCH_DIR", tmp_path)
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n")
    patch_file = tmp_path / "patches.jsonl"
    patch_file.write_text('{"instance_id":"owner__repo-1","model_patch":"model patch"}\n')
    cfg = SweBenchGenerationConfig(
        _init_nested=True,
        input_file=str(dataset),
        output_file=str(tmp_path / "output.jsonl"),
        agent_framework=framework,
        model_patch_file=str(patch_file) if framework == "model_patch" else None,
        multilingual=True,
        execution_backend="opensandbox",
        server=OmegaConf.create({"base_url": "http://unused.example:8000/v1"}),
    )

    task = SweBenchGenerationTask(cfg)

    task._execute_local_command.assert_not_called()
    detect.assert_not_called()
    assert task.cfg.opensandbox_proxy_host is None
    assert "SWE-bench" not in task.opensandbox_setup_commands["agent"]
    assert "SWE-bench" in task.opensandbox_setup_commands["eval"]


@pytest.mark.parametrize(
    "framework,patch,patch_directory",
    [
        ("gold_patch", "gold patch", "gold_patches"),
        ("model_patch", "model patch", "model_patches"),
    ],
)
@pytest.mark.parametrize(
    "dataset_type,evaluator_entrypoint,evaluator_argument",
    [
        (
            SupportedDatasetTypes.swe_bench,
            "swebench.harness.run_local_evaluation",
            "--dataset_name",
        ),
        (
            SupportedDatasetTypes.swe_bench_pro,
            "swebench.harness.run_local_evaluation",
            "--raw_sample_path",
        ),
        (
            SupportedDatasetTypes.swe_rebench_v2,
            "scripts/local_eval.py",
            "--json",
        ),
    ],
)
def test_supplied_patch_runs_only_remote_evaluation(
    executor,
    monkeypatch,
    tmp_path,
    framework,
    patch,
    patch_directory,
    dataset_type,
    evaluator_entrypoint,
    evaluator_argument,
):
    instance_id = "owner__repo-1"
    data_point = {
        "instance_id": instance_id,
        "docker_image": "swebench/sweb.eval.x86_64.owner_1776_repo-1",
        "problem_statement": "Fix the bug",
        "patch": "gold patch",
        "test_patch": "hidden test patch",
    }
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text(json.dumps(data_point) + "\n")
    patch_file = tmp_path / "patches.jsonl"
    patch_file.write_text(json.dumps({"instance_id": instance_id, "model_patch": "model patch"}) + "\n")
    report = {
        instance_id: {
            "resolved": True,
            "patch_exists": True,
            "patch_successfully_applied": True,
        }
    }
    sandbox, create = _sandbox(
        executor,
        monkeypatch,
        _archive(
            "report.json",
            json.dumps(report).encode(),
        ),
    )
    detect = MagicMock(side_effect=AssertionError("supplied patches do not need a model proxy"))
    monkeypatch.setattr(executor, "detect_proxy_host", detect)
    monkeypatch.setattr(SweBenchGenerationTask, "_execute_local_command", AsyncMock())
    monkeypatch.setattr("nemo_skills.inference.eval.opensandbox.OpenSandboxExecutor", lambda **kwargs: executor)
    monkeypatch.setattr("nemo_skills.inference.eval.swebench.SCRATCH_DIR", tmp_path)
    cfg = SweBenchGenerationConfig(
        _init_nested=True,
        input_file=str(dataset),
        output_file=str(tmp_path / "results/output.jsonl"),
        agent_framework=framework,
        model_patch_file=str(patch_file) if framework == "model_patch" else None,
        dataset_type=dataset_type,
        execution_backend="opensandbox",
        server=OmegaConf.create({"base_url": "http://unused.example:8000/v1"}),
    )
    task = SweBenchGenerationTask(cfg)
    task.rollout_semaphore = asyncio.Semaphore(1)
    task.eval_semaphore = asyncio.Semaphore(1)

    output = asyncio.run(task.process_single_datapoint(data_point, []))

    assert output["swe-bench-metrics"]["resolved"] is True
    assert output["swe-bench-outputs"]["model_patch"] == patch
    prediction = tmp_path / f"results/{patch_directory}/{instance_id}.jsonl"
    assert prediction.is_file()
    assert create.await_count == 1
    assert create.call_args.kwargs["metadata"]["mode"] == "eval"
    detect.assert_not_called()
    uploaded_paths = [call.args[0] for call in sandbox.files.write_file.call_args_list]
    assert uploaded_paths == [f"/predictions_mount/{instance_id}.jsonl", "/input_mount/dataset.jsonl"]
    commands = "\n".join(call.args[0] for call in sandbox.commands.run.call_args_list)
    assert evaluator_entrypoint in commands
    assert evaluator_argument in commands
    assert (tmp_path / f"results/opensandbox_logs/{instance_id}_eval.log").is_file()
    assert not (tmp_path / f"results/opensandbox_logs/{instance_id}_agent.log").exists()


@pytest.mark.parametrize("framework", ["gold_patch", "model_patch"])
def test_supplied_patch_keeps_opensandbox_dataset_restrictions(executor, monkeypatch, tmp_path, framework):
    monkeypatch.setattr("nemo_skills.inference.eval.opensandbox.OpenSandboxExecutor", lambda **kwargs: executor)
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n")
    cfg = SweBenchGenerationConfig(
        _init_nested=True,
        input_file=str(dataset),
        output_file=str(tmp_path / "output.jsonl"),
        agent_framework=framework,
        model_patch_file=str(tmp_path / "patches.jsonl") if framework == "model_patch" else None,
        dataset_type=SupportedDatasetTypes.swe_bench,
        swe_zero_container="docker://swe-zero",
        execution_backend="opensandbox",
    )

    with pytest.raises(ValueError, match="SWE-Zero"):
        SweBenchGenerationTask(cfg)


def test_swe_bench_pro_v2_uses_opensandbox_without_eval_harness(executor, monkeypatch, tmp_path):
    monkeypatch.setattr(executor, "detect_proxy_host", MagicMock(return_value="10.0.0.3"))
    monkeypatch.setattr(SweBenchGenerationTask, "_execute_local_command", AsyncMock())
    monkeypatch.setattr("nemo_skills.inference.eval.opensandbox.OpenSandboxExecutor", lambda **kwargs: executor)
    monkeypatch.setattr("nemo_skills.inference.eval.swebench.SCRATCH_DIR", tmp_path)
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_text("{}\n")
    cfg = SweBenchGenerationConfig(
        _init_nested=True,
        input_file=str(dataset),
        output_file=str(tmp_path / "output.jsonl"),
        agent_framework="mini_swe_agent",
        dataset_type=SupportedDatasetTypes.swe_bench_pro_v2,
        execution_backend="opensandbox",
        server=OmegaConf.create({"base_url": "http://gpu.example:8000/v1"}),
    )

    task = SweBenchGenerationTask(cfg)

    task._execute_local_command.assert_not_called()
    assert "mini-swe-agent" in task.opensandbox_setup_commands["agent"]
    assert "SWE-bench" not in task.opensandbox_setup_commands["eval"]
    assert "astral.sh/uv" not in task.opensandbox_setup_commands["eval"]


@pytest.mark.parametrize(
    "dataset_type,stage_eval_harness",
    [
        (SupportedDatasetTypes.swe_bench, True),
        (SupportedDatasetTypes.swe_bench_pro_v2, False),
    ],
)
@pytest.mark.parametrize("instance_id", ["owner__repo-1", "axios__axios-4731"])
def test_backend_dispatch_preserves_native_artifact_contract(
    executor, monkeypatch, tmp_path, instance_id, dataset_type, stage_eval_harness
):
    sandbox, _ = _sandbox(executor, monkeypatch, _archive())
    args = _arguments(tmp_path)
    args["data_point"]["instance_id"] = instance_id
    args["mode"] = "eval"
    task = object.__new__(SweBenchGenerationTask)
    task.cfg = SimpleNamespace(
        input_file=str(args["input_file"]),
        setup_timeout=args["setup_timeout"],
        agent_framework="mini_swe_agent",
        dataset_type=dataset_type,
        opensandbox_proxy_host=args["proxy_host"],
        block_network=False,
    )
    task.opensandbox_executor = executor
    task.opensandbox_setup_commands = {"eval": "install-runtime"}
    task.output_dir = args["output_dir"]
    _configure_task_scratch(task, tmp_path)
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
    setup_command = sandbox.commands.run.call_args_list[0].args[0]
    assert ("SWE-bench" in setup_command) is stage_eval_harness
    sandbox.kill.assert_awaited_once()


def test_remote_evaluation_failure_returns_failed_patch_metrics(tmp_path):
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

    output = asyncio.run(run())

    assert output["swe-bench-metrics"] == {
        "resolved": False,
        "patch_exists": True,
        "patch_successfully_applied": False,
    }


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
        "opencode": ["opencode", "node"],
        "claude_code": ["claude-code", "node"],
    }[framework]
    for path in paths:
        assert f"cp -a /root/{path} /root_mount/" in commands
    if framework == "opencode":
        assert "PATH=/root_mount/opencode/bin:/root_mount/node/bin" in commands
    if framework == "claude_code":
        assert "PATH=/root_mount/claude-code/bin:/root_mount/node/bin" in commands
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
