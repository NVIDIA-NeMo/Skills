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

"""Unit tests for ``nemo_skills.pipeline.utils.ray_executor``.

The tests run against a real Ray cluster started inside the test file
(``ray.init`` in a module-scoped fixture). No GPU is required — Ray runs
local-only with two CPU slots — and the cluster lifecycle is fully managed
here so the tests are self-contained.

Behavioural assertions submit tiny Python entrypoints through
``RayJobClient`` and inspect the resulting ``JobInfo`` once Ray reports a
terminal state. Error-path tests monkey-patch a single method on the live
``JobSubmissionClient`` so we can exercise the wrapper's defensive code
without resorting to module-level import-time SDK mocks.

Two configuration-parsing tests at the end do not touch Ray at all — they
just assert that ``get_ray_client`` reads ``cluster_config["ray"]`` keys
correctly — so they remain plain unit tests.
"""

import time

import pytest

# Skip the entire module if Ray is not installed in this environment. CI
# pulls ray[default] in via the dev extras (see requirements/common-tests.txt);
# contributors running `pytest` without dev extras get a clean skip rather
# than an ImportError.
ray = pytest.importorskip("ray")
pytest.importorskip("ray.job_submission")

from nemo_skills.pipeline.utils.ray_executor import (  # noqa: E402
    RayJobClient,
    RayJobConfig,
    get_ray_client,
)

# Treat each individual test as bounded by the wait helpers below; this
# module-level marker prevents an unexpected stall from hanging CI.
pytestmark = pytest.mark.timeout(120)


# ---------------------------------------------------------------------------
# Cluster + client fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _ray_cluster():
    """Start a local Ray cluster (head + workers in this process)."""
    if ray.is_initialized():
        ray.shutdown()
    ray.init(
        num_cpus=2,
        include_dashboard=True,
        ignore_reinit_error=True,
        configure_logging=False,
        log_to_driver=False,
    )
    try:
        yield
    finally:
        ray.shutdown()


@pytest.fixture
def client(_ray_cluster):
    """A ``RayJobClient`` connected to the in-process Ray cluster."""
    return RayJobClient(ray_address="auto", namespace="nemo-skills-tests")


def _wait_until_terminal(client: RayJobClient, job_id: str, timeout: float = 60.0) -> str:
    """Poll the live cluster until ``job_id`` reaches a terminal state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        status = client.get_job_status(job_id)
        if any(t in status for t in ("SUCCEEDED", "FAILED", "STOPPED")):
            return status
        time.sleep(0.5)
    raise AssertionError(f"job {job_id} did not reach a terminal state within {timeout}s")


def _raiser(exc: BaseException):
    """Return a callable that raises ``exc`` regardless of arguments."""

    def _fn(*args, **kwargs):
        raise exc

    return _fn


# ---------------------------------------------------------------------------
# submit_job — round-trip behaviour against a real Ray cluster
# ---------------------------------------------------------------------------


def test_submit_job_returns_submission_id(client, tmp_path):
    """``submit_job`` returns the same submission ID Ray uses internally.

    This also serves as the regression for the Ray 2.54 ``submission_id=``
    kwarg fix: if a future refactor reverts to the deprecated ``job_id=``
    kwarg, ``client.submit_job`` raises ``TypeError`` immediately — the test
    fails at submission time rather than silently regressing.
    """
    config = RayJobConfig(
        name="ns-test-id",
        command="python -c \"print('hello')\"",
        num_gpus=0,
        num_cpus=1,
        log_dir=str(tmp_path),
    )

    sub_id = client.submit_job(config)
    try:
        assert sub_id == "ns-test-id"
        status = _wait_until_terminal(client, sub_id)
        assert "SUCCEEDED" in status
    finally:
        client.cancel_job(sub_id)


def test_submit_job_env_vars_reach_the_worker(client, tmp_path):
    """``env_vars`` from RayJobConfig must be visible inside the entrypoint."""
    expected = "ns-real-ray-marker-value"
    out_file = tmp_path / "env_marker.txt"
    config = RayJobConfig(
        name="ns-test-env",
        command=f"python -c \"import os; open({str(out_file)!r}, 'w').write(os.environ['NS_TEST_VAR'])\"",
        num_gpus=0,
        num_cpus=1,
        env_vars={"NS_TEST_VAR": expected},
        log_dir=str(tmp_path),
    )

    sub_id = client.submit_job(config)
    try:
        status = _wait_until_terminal(client, sub_id)
        assert "SUCCEEDED" in status, f"job ended in {status}; logs:\n{client.get_job_logs(sub_id)}"
        assert out_file.read_text() == expected
    finally:
        client.cancel_job(sub_id)


def test_submit_job_preserves_runtime_env_overrides(client, tmp_path):
    """Pre-existing ``runtime_env`` keys (e.g., ``env_vars``) must merge, not clobber."""
    out_file = tmp_path / "merge_marker.txt"
    config = RayJobConfig(
        name="ns-test-merge",
        # Both BASE_VAR (pre-set in runtime_env) and OVERLAY_VAR (added via env_vars)
        # should land in the worker's environment.
        command=(
            f"python -c \"import os; open({str(out_file)!r}, 'w').write("
            "os.environ.get('BASE_VAR', '') + ':' + os.environ.get('OVERLAY_VAR', ''))\""
        ),
        num_gpus=0,
        num_cpus=1,
        env_vars={"OVERLAY_VAR": "from_env_vars"},
        runtime_env={"env_vars": {"BASE_VAR": "from_runtime_env"}},
        log_dir=str(tmp_path),
    )

    sub_id = client.submit_job(config)
    try:
        status = _wait_until_terminal(client, sub_id)
        assert "SUCCEEDED" in status
        assert out_file.read_text() == "from_runtime_env:from_env_vars"
    finally:
        client.cancel_job(sub_id)


def test_submit_job_creates_log_dir(client, tmp_path):
    """``log_dir`` must be created if it does not already exist."""
    log_dir = tmp_path / "deeply" / "nested" / "ray_jobs"
    assert not log_dir.exists()

    config = RayJobConfig(
        name="ns-test-logdir",
        command='python -c "pass"',
        num_gpus=0,
        num_cpus=1,
        log_dir=str(log_dir),
    )

    sub_id = client.submit_job(config)
    try:
        assert log_dir.is_dir()
        _wait_until_terminal(client, sub_id)
    finally:
        client.cancel_job(sub_id)


# ---------------------------------------------------------------------------
# _wait_for_dependencies — real success / failure / timeout
# ---------------------------------------------------------------------------


def test_wait_for_dependencies_returns_on_succeeded(client, tmp_path):
    """A real successful dependency lets the wait return cleanly."""
    dep_config = RayJobConfig(
        name="ns-test-dep-ok",
        command="python -c \"print('dep ok')\"",
        num_gpus=0,
        num_cpus=1,
        log_dir=str(tmp_path),
    )
    dep_id = client.submit_job(dep_config)
    try:
        _wait_until_terminal(client, dep_id)
        # Short poll to keep the test fast; the dependency is already SUCCEEDED.
        client._wait_for_dependencies([dep_id], poll_interval=1, timeout=30)
    finally:
        client.cancel_job(dep_id)


@pytest.mark.parametrize("exit_code", [1, 2])
def test_wait_for_dependencies_raises_on_terminal_failure(client, tmp_path, exit_code):
    """A real failed dependency raises ``RuntimeError`` describing the status."""
    dep_config = RayJobConfig(
        name=f"ns-test-dep-fail-{exit_code}",
        command=f'python -c "import sys; sys.exit({exit_code})"',
        num_gpus=0,
        num_cpus=1,
        log_dir=str(tmp_path),
    )
    dep_id = client.submit_job(dep_config)
    try:
        _wait_until_terminal(client, dep_id)
        with pytest.raises(RuntimeError, match="FAILED"):
            client._wait_for_dependencies([dep_id], poll_interval=1, timeout=30)
    finally:
        client.cancel_job(dep_id)


def test_wait_for_dependencies_raises_on_timeout(client, tmp_path):
    """If a dependency never completes within ``timeout``, ``TimeoutError`` fires."""
    # Long-running job: sleeps longer than the wait budget.
    dep_config = RayJobConfig(
        name="ns-test-dep-timeout",
        command='python -c "import time; time.sleep(60)"',
        num_gpus=0,
        num_cpus=1,
        log_dir=str(tmp_path),
    )
    dep_id = client.submit_job(dep_config)
    try:
        with pytest.raises(TimeoutError, match=dep_id):
            client._wait_for_dependencies([dep_id], poll_interval=1, timeout=2)
    finally:
        client.cancel_job(dep_id)


# ---------------------------------------------------------------------------
# get_job_status / get_job_logs — round-trip + error swallowing
# ---------------------------------------------------------------------------


def test_get_job_status_stringifies(client, tmp_path):
    """``get_job_status`` returns a plain ``str`` even when Ray returns an enum."""
    config = RayJobConfig(
        name="ns-test-status",
        command="python -c \"print('ok')\"",
        num_gpus=0,
        num_cpus=1,
        log_dir=str(tmp_path),
    )
    sub_id = client.submit_job(config)
    try:
        _wait_until_terminal(client, sub_id)
        status = client.get_job_status(sub_id)
        assert isinstance(status, str)
        assert status == "SUCCEEDED"
    finally:
        client.cancel_job(sub_id)


def test_get_job_logs_returns_underlying_logs_on_success(client, tmp_path):
    """``get_job_logs`` returns the entrypoint's captured stdout/stderr."""
    sentinel = "ns-real-ray-stdout-sentinel"
    config = RayJobConfig(
        name="ns-test-logs",
        command=f"python -c \"print('{sentinel}')\"",
        num_gpus=0,
        num_cpus=1,
        log_dir=str(tmp_path),
    )
    sub_id = client.submit_job(config)
    try:
        _wait_until_terminal(client, sub_id)
        logs = client.get_job_logs(sub_id)
        assert sentinel in logs
    finally:
        client.cancel_job(sub_id)


def test_get_job_logs_returns_empty_string_on_error(client, monkeypatch, caplog):
    """When the underlying ``get_job_logs`` raises, return ``""`` and warn."""
    monkeypatch.setattr(client.client, "get_job_logs", _raiser(RuntimeError("connection lost")))

    with caplog.at_level("WARNING", logger="nemo_skills.pipeline.utils.ray_executor"):
        result = client.get_job_logs("nonexistent-job")

    assert result == ""
    assert any("connection lost" in rec.message for rec in caplog.records), (
        "expected a WARNING log naming the underlying error"
    )


def test_get_job_logs_propagates_unexpected_errors(client, monkeypatch):
    """Programmer errors (``ValueError``, ``KeyError``, ...) must NOT be swallowed."""
    monkeypatch.setattr(client.client, "get_job_logs", _raiser(ValueError("bad arg")))

    with pytest.raises(ValueError, match="bad arg"):
        client.get_job_logs("nonexistent-job")


# ---------------------------------------------------------------------------
# cancel_job / list_jobs — round-trip + error swallowing
# ---------------------------------------------------------------------------


def test_cancel_job_swallows_error_and_logs_warning(client, monkeypatch, caplog):
    """When the underlying ``stop_job`` raises, log a warning and return ``None``."""
    monkeypatch.setattr(client.client, "stop_job", _raiser(RuntimeError("already stopped")))

    with caplog.at_level("WARNING", logger="nemo_skills.pipeline.utils.ray_executor"):
        # Should not raise.
        client.cancel_job("any-job")

    assert any("already stopped" in rec.message for rec in caplog.records)


def test_list_jobs_returns_list_against_real_cluster(client):
    """``list_jobs`` returns a list (possibly empty) against the live cluster."""
    jobs = client.list_jobs()
    assert isinstance(jobs, list)


def test_list_jobs_returns_empty_list_on_error(client, monkeypatch):
    """When the underlying ``list_jobs`` raises, return ``[]``."""
    monkeypatch.setattr(client.client, "list_jobs", _raiser(RuntimeError("api unavailable")))

    assert client.list_jobs() == []


# ---------------------------------------------------------------------------
# get_ray_client factory — pure config parsing, no Ray cluster needed
# ---------------------------------------------------------------------------


def test_get_ray_client_reads_address_and_namespace(monkeypatch):
    """``get_ray_client`` reads ``ray.address`` / ``ray.namespace`` from config."""
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
    """``get_ray_client`` defaults ``ray_address`` to "auto" and namespace to "nemo"."""
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


# ---------------------------------------------------------------------------
# add_task() Ray-branch resource-scaling regression tests.
#
# These don't touch a live Ray cluster — they monkey-patch the Ray client and
# env-var lookup so add_task() goes down the executor=="ray" branch and the
# resulting RayJobConfig can be inspected.
# ---------------------------------------------------------------------------


def _run_add_task_ray(monkeypatch, *, num_gpus, num_nodes):
    """Drive ``add_task`` down its Ray branch and return the submitted ``RayJobConfig``."""
    from nemo_skills.pipeline.utils import exp as exp_module

    captured = {}

    class _FakeRayClient:
        def submit_job(self, config):
            captured["config"] = config
            return "fake-submission-id"

    monkeypatch.setattr(
        "nemo_skills.pipeline.utils.exp.get_ray_client",
        lambda cluster_config: _FakeRayClient(),
    )
    monkeypatch.setattr(
        "nemo_skills.pipeline.utils.exp.get_env_variables",
        lambda cluster_config: {"HF_HOME": "/mnt/hf_home"},
    )

    cluster_config = {
        "executor": "ray",
        "ray": {"address": "auto"},
        "skip_hf_home_check": True,
    }

    submission_id = exp_module.add_task(
        exp=None,
        cmd="echo hello",
        task_name="ray-resource-scaling-test",
        cluster_config=cluster_config,
        container="dummy-container",
        num_gpus=num_gpus,
        num_nodes=num_nodes,
    )

    assert submission_id == "fake-submission-id"
    return captured["config"]


def test_add_task_ray_scales_num_gpus_by_num_nodes(monkeypatch):
    """Multi-node Ray jobs must request ``num_gpus * num_nodes`` total GPUs.

    ``add_task``'s ``num_gpus`` parameter is per-node (matches ``get_executor``'s
    ``int(gpus_per_node) * num_nodes`` scaling in the same file).
    ``RayJobConfig.num_gpus`` is per-job total — ``ray_executor.py`` divides by
    ``num_nodes`` to derive ``entrypoint_num_gpus``. Without the multiplier,
    a 2-GPU/node × 2-node job silently requests only 2 total GPUs.
    """
    config = _run_add_task_ray(monkeypatch, num_gpus=2, num_nodes=2)
    assert config.num_gpus == 4, (
        f"Expected num_gpus=4 (2 GPUs/node × 2 nodes); got {config.num_gpus}. "
        "RayJobConfig.num_gpus is per-job total; add_task num_gpus is per-node."
    )
    # Sanity: num_cpus already scales by num_nodes; num_nodes propagates.
    assert config.num_cpus == 16  # default 8 CPUs/node × 2 nodes
    assert config.num_nodes == 2


def test_add_task_ray_default_num_gpus_scales_by_num_nodes(monkeypatch):
    """When ``num_gpus`` is None the default is 1/node, so total = ``num_nodes``."""
    config = _run_add_task_ray(monkeypatch, num_gpus=None, num_nodes=3)
    assert config.num_gpus == 3, f"Expected num_gpus=3 (default 1/node × 3 nodes); got {config.num_gpus}."
    assert config.num_nodes == 3


def test_add_task_ray_single_node_unchanged(monkeypatch):
    """Single-node Ray jobs are unaffected: ``num_gpus * 1 == num_gpus``."""
    config = _run_add_task_ray(monkeypatch, num_gpus=8, num_nodes=1)
    assert config.num_gpus == 8
    assert config.num_nodes == 1
