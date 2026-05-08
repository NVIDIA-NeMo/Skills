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

"""Mock-based unit tests for `nemo_skills.pipeline.utils.ray_executor` client layer.

These tests exercise the Ray submission client without requiring an actual Ray
cluster. The Ray Python SDK is mocked at import time so the module can be
loaded in environments where `ray` is not installed.

Coverage (per the staged Ray PR scope):
- submit_job builds runtime_env correctly from RayJobConfig
- env_vars from config merge into runtime_env["env_vars"]
- pre-existing runtime_env["env_vars"] is preserved when merging
- _wait_for_dependencies returns cleanly when status reaches SUCCEEDED
- _wait_for_dependencies raises on FAILED / STOPPED
- _wait_for_dependencies raises TimeoutError after timeout elapses
- get_job_logs returns "" + warns on underlying client errors
- cancel_job swallows and warns on underlying client errors
- submit_job uses `submission_id=` (not the deprecated `job_id=` kwarg)
- get_ray_client factory reads ray.address / ray.namespace correctly
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


# ----------------------------------------------------------------------------
# Mock the Ray SDK at import time so this test file works in environments where
# `ray` is not installed (e.g., the NeMo-Skills CI without GPU/Ray deps).
# ----------------------------------------------------------------------------
def _install_ray_mocks():
    ray_module = MagicMock(name="ray")
    ray_module.is_initialized = MagicMock(return_value=False)
    ray_module.init = MagicMock()
    ray_module.cluster_resources = MagicMock(return_value={"CPU": 8.0, "GPU": 1.0})

    job_submission_module = MagicMock(name="ray.job_submission")
    job_submission_module.JobSubmissionClient = MagicMock(name="JobSubmissionClient")

    sys.modules["ray"] = ray_module
    sys.modules["ray.job_submission"] = job_submission_module
    return ray_module, job_submission_module


_install_ray_mocks()

# Import after mocks are in place.
from nemo_skills.pipeline.utils.ray_executor import (  # noqa: E402
    RayJobClient,
    RayJobConfig,
    get_ray_client,
)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _make_client(client_mock=None) -> RayJobClient:
    """Construct a RayJobClient with a mocked JobSubmissionClient instance."""
    client = RayJobClient.__new__(RayJobClient)
    client.ray_address = "auto"
    client.namespace = "nemo"
    client.client = client_mock if client_mock is not None else MagicMock()
    return client


# ----------------------------------------------------------------------------
# submit_job — runtime_env construction
# ----------------------------------------------------------------------------
def test_submit_job_uses_submission_id_not_job_id(tmp_path):
    """Regression: Ray 2.54 deprecated `job_id=`; we must use `submission_id=`.

    If a future refactor accidentally reverts to `job_id=`, this test catches it.
    """
    underlying = MagicMock()
    underlying.submit_job.return_value = "ray_sub_123"
    client = _make_client(underlying)
    config = RayJobConfig(
        name="my_job",
        command="python train.py",
        num_gpus=1,
        num_cpus=8,
        num_nodes=1,
        log_dir=str(tmp_path),
    )

    sub_id = client.submit_job(config)

    assert sub_id == "ray_sub_123"
    underlying.submit_job.assert_called_once()
    kwargs = underlying.submit_job.call_args.kwargs
    assert "submission_id" in kwargs, "must use submission_id (not the deprecated job_id)"
    assert "job_id" not in kwargs
    assert kwargs["submission_id"] == "my_job"


def test_submit_job_builds_runtime_env_from_env_vars(tmp_path):
    underlying = MagicMock()
    underlying.submit_job.return_value = "ray_sub_456"
    client = _make_client(underlying)
    config = RayJobConfig(
        name="env_job",
        command="echo hi",
        env_vars={"HF_HOME": "/models/hf", "TOKENIZERS_PARALLELISM": "false"},
        log_dir=str(tmp_path),
    )

    client.submit_job(config)

    kwargs = underlying.submit_job.call_args.kwargs
    assert kwargs["runtime_env"]["env_vars"] == {
        "HF_HOME": "/models/hf",
        "TOKENIZERS_PARALLELISM": "false",
    }


def test_submit_job_preserves_preexisting_runtime_env_overrides(tmp_path):
    """User-supplied runtime_env (e.g., working_dir, pip) must not be clobbered."""
    underlying = MagicMock()
    underlying.submit_job.return_value = "ray_sub_789"
    client = _make_client(underlying)
    config = RayJobConfig(
        name="merge_job",
        command="python -m pkg.main",
        env_vars={"MY_VAR": "1"},
        log_dir=str(tmp_path),
        runtime_env={"working_dir": "/repo", "pip": ["numpy==1.26"]},
    )

    client.submit_job(config)

    runtime_env = underlying.submit_job.call_args.kwargs["runtime_env"]
    assert runtime_env["working_dir"] == "/repo"
    assert runtime_env["pip"] == ["numpy==1.26"]
    assert runtime_env["env_vars"] == {"MY_VAR": "1"}


def test_submit_job_per_node_resource_split(tmp_path):
    """num_gpus / num_cpus get divided across num_nodes for entrypoint resources."""
    underlying = MagicMock()
    underlying.submit_job.return_value = "ray_sub_split"
    client = _make_client(underlying)
    config = RayJobConfig(
        name="multi_node",
        command="echo multi",
        num_gpus=8,
        num_cpus=64,
        num_nodes=2,
        log_dir=str(tmp_path),
    )

    client.submit_job(config)

    kwargs = underlying.submit_job.call_args.kwargs
    assert kwargs["entrypoint_num_gpus"] == 4.0  # 8 / 2
    assert kwargs["entrypoint_num_cpus"] == 32.0  # 64 / 2


def test_submit_job_creates_log_dir(tmp_path):
    """The log_dir directory must be created if it does not exist."""
    log_dir = tmp_path / "deeply" / "nested" / "ray_jobs"
    assert not log_dir.exists()
    underlying = MagicMock()
    underlying.submit_job.return_value = "ray_sub_logdir"
    client = _make_client(underlying)
    config = RayJobConfig(
        name="log_dir_job",
        command="echo x",
        log_dir=str(log_dir),
    )

    client.submit_job(config)

    assert log_dir.is_dir()


# ----------------------------------------------------------------------------
# _wait_for_dependencies — happy / error / timeout
# ----------------------------------------------------------------------------
def test_wait_for_dependencies_returns_on_succeeded():
    underlying = MagicMock()
    underlying.get_job_status.side_effect = ["RUNNING", "SUCCEEDED"]
    client = _make_client(underlying)

    # poll_interval=0 to avoid sleeping in tests
    with patch("nemo_skills.pipeline.utils.ray_executor.time.sleep"):
        client._wait_for_dependencies(["dep_job_1"], poll_interval=0, timeout=60)

    assert underlying.get_job_status.call_count == 2


@pytest.mark.parametrize("terminal_status", ["FAILED", "STOPPED"])
def test_wait_for_dependencies_raises_on_terminal_failure(terminal_status):
    underlying = MagicMock()
    underlying.get_job_status.return_value = terminal_status
    client = _make_client(underlying)

    with patch("nemo_skills.pipeline.utils.ray_executor.time.sleep"):
        with pytest.raises(RuntimeError, match=terminal_status):
            client._wait_for_dependencies(["bad_dep"], poll_interval=0, timeout=60)


def test_wait_for_dependencies_raises_on_timeout():
    underlying = MagicMock()
    underlying.get_job_status.return_value = "RUNNING"
    client = _make_client(underlying)

    # Patch time.time to return ever-increasing values so the timeout check
    # fires after one iteration. Use a small timeout to keep the test fast.
    fake_clock = iter([0.0, 100.0, 200.0])

    def _fake_time():
        return next(fake_clock)

    with patch("nemo_skills.pipeline.utils.ray_executor.time.time", side_effect=_fake_time):
        with patch("nemo_skills.pipeline.utils.ray_executor.time.sleep"):
            with pytest.raises(TimeoutError, match="dep_timeout"):
                client._wait_for_dependencies(["dep_timeout"], poll_interval=0, timeout=10)


# ----------------------------------------------------------------------------
# get_job_logs / cancel_job — error swallowing
# ----------------------------------------------------------------------------
def test_get_job_logs_returns_empty_string_on_error(caplog):
    underlying = MagicMock()
    underlying.get_job_logs.side_effect = RuntimeError("connection lost")
    client = _make_client(underlying)

    with caplog.at_level("WARNING", logger="nemo_skills.pipeline.utils.ray_executor"):
        result = client.get_job_logs("some_job")

    assert result == ""
    assert any("connection lost" in rec.message for rec in caplog.records), \
        "expected a WARNING log naming the underlying error"


def test_cancel_job_swallows_error_and_logs_warning(caplog):
    underlying = MagicMock()
    underlying.stop_job.side_effect = RuntimeError("already stopped")
    client = _make_client(underlying)

    with caplog.at_level("WARNING", logger="nemo_skills.pipeline.utils.ray_executor"):
        # Should not raise.
        client.cancel_job("some_job")

    underlying.stop_job.assert_called_once_with("some_job")
    assert any("already stopped" in rec.message for rec in caplog.records), \
        "expected a WARNING log naming the underlying error"


def test_get_job_logs_returns_underlying_logs_on_success():
    underlying = MagicMock()
    underlying.get_job_logs.return_value = "stdout/stderr captured"
    client = _make_client(underlying)

    assert client.get_job_logs("ok_job") == "stdout/stderr captured"


def test_get_job_status_stringifies():
    """get_job_status must coerce Ray's status enum to a plain string."""
    underlying = MagicMock()
    underlying.get_job_status.return_value = "SUCCEEDED"
    client = _make_client(underlying)

    assert client.get_job_status("any") == "SUCCEEDED"


def test_list_jobs_returns_empty_list_on_error():
    underlying = MagicMock()
    underlying.list_jobs.side_effect = RuntimeError("api unavailable")
    client = _make_client(underlying)

    assert client.list_jobs() == []


# ----------------------------------------------------------------------------
# get_ray_client factory
# ----------------------------------------------------------------------------
def test_get_ray_client_reads_address_and_namespace(monkeypatch):
    """Factory reads `ray.address` and `ray.namespace` from cluster_config."""
    # Patch the RayJobClient constructor to capture args without needing a live cluster.
    captured = {}

    class _DummyClient:
        def __init__(self, ray_address: str, namespace: str):
            captured["ray_address"] = ray_address
            captured["namespace"] = namespace

    monkeypatch.setattr(
        "nemo_skills.pipeline.utils.ray_executor.RayJobClient",
        _DummyClient,
    )

    cluster_config = {
        "executor": "ray",
        "ray": {"address": "ray://10.0.0.1:10001", "namespace": "team-a"},
    }
    get_ray_client(cluster_config)

    assert captured["ray_address"] == "ray://10.0.0.1:10001"
    assert captured["namespace"] == "team-a"


def test_get_ray_client_uses_defaults_when_ray_block_absent(monkeypatch):
    captured = {}

    class _DummyClient:
        def __init__(self, ray_address: str, namespace: str):
            captured["ray_address"] = ray_address
            captured["namespace"] = namespace

    monkeypatch.setattr(
        "nemo_skills.pipeline.utils.ray_executor.RayJobClient",
        _DummyClient,
    )

    get_ray_client({"executor": "ray"})  # no `ray:` block at all

    assert captured["ray_address"] == "auto"
    assert captured["namespace"] == "nemo"
