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

"""Initial Ray support for NeMo-Skills.

This module provides two layers that together let the pipeline target a
top-level Ray scheduler (standalone Ray clusters and Ray-on-Slurm), distinct
from ``nemo_run.core.execution.kuberay.KubeRayExecutor`` which targets
Kubernetes-managed Ray:

1. ``RayJobConfig`` + ``RayJobClient`` + ``get_ray_client`` — low-level Ray
   Jobs API client used to submit jobs, chain dependencies by submission ID,
   and read status/logs.

2. ``RayExecutor`` — a nemo-run Executor dataclass that holds Ray cluster
   parameters. Routing in ``get_executor()`` / ``add_task()`` bridges from
   ``cluster_config["executor"] == "ray"`` to the client above instead of
   ``nemo_run.Experiment.add()``.

Supported in this release:
- Single-command Ray jobs (SFT, eventually GRPO)
- Dependency chaining by Ray submission IDs
- Shared-FS runtime/code visibility

Out of scope (the routing layer raises ``NotImplementedError``):
- Sandbox judge containers
- Server co-scheduling (vLLM/SGLang lifecycle alongside main job)
- Heterogeneous tasks
- Multi-command task groups
- Generic eval/generate/server orchestration
"""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from nemo_run.core.execution.base import Executor

LOG = logging.getLogger(__name__)

# Exception types that Ray's job-submission client may surface for network and
# HTTP-level failures. ``OSError`` covers stdlib + requests-based connection
# and timeout errors (``ConnectionError``, ``TimeoutError``, and
# ``requests.exceptions.RequestException`` are all ``OSError`` subclasses in
# Python 3); ``RuntimeError`` is what Ray's dashboard client wraps non-2xx HTTP
# responses in. Used to bound the scope of ``try`` blocks around Ray client
# calls so unrelated programmer errors (``ValueError``, ``KeyError``, …)
# propagate naturally instead of being swallowed.
_RAY_CLIENT_EXCEPTIONS = (OSError, RuntimeError)


# ---------------------------------------------------------------------------
# Ray Jobs API client layer
# ---------------------------------------------------------------------------


def _import_ray():
    """Lazy-load Ray + JobSubmissionClient.

    Kept out of module-level imports so callers can construct ``RayJobConfig``
    or hold typed references to ``RayJobClient`` without ray installed —
    only actual cluster connection requires it.
    """
    try:
        import ray
        from ray.job_submission import JobSubmissionClient
    except ImportError as e:
        raise ImportError("ray is required for Ray executor. Install with: pip install ray") from e
    return ray, JobSubmissionClient


@dataclass
class RayJobConfig:
    """Configuration for a Ray job submission."""

    name: str
    command: str
    num_gpus: int = 1
    num_cpus: int = 8
    num_nodes: int = 1
    env_vars: Optional[Dict[str, str]] = None
    log_dir: str = "/tmp/ray_jobs"
    dependencies: Optional[List[str]] = None  # Job submission IDs to wait for
    runtime_env: Optional[Dict[str, Any]] = None


class RayJobClient:
    """Client to submit and manage jobs on Ray cluster."""

    def __init__(self, ray_address: str = "auto", namespace: str = "nemo"):
        """
        Initialize Ray cluster connection.

        Args:
            ray_address: Ray cluster address (e.g., "ray://127.0.0.1:10001" or "auto")
            namespace: Ray namespace for job isolation
        """
        self.ray_address = ray_address
        self.namespace = namespace
        self.client = None
        self._connect()

    def _connect(self):
        """Connect to Ray cluster.

        On success: stores the client on ``self.client`` and returns it.
        On failure: raises and ``self.client`` is left unchanged.
        """
        ray, JobSubmissionClient = _import_ray()
        try:
            if not ray.is_initialized():
                ray.init(address=self.ray_address, namespace=self.namespace, ignore_reinit_error=True)
            self.client = JobSubmissionClient(address=self.ray_address)
            LOG.info("Connected to Ray cluster at %s", self.ray_address)

            # Get cluster info
            cluster_info = ray.cluster_resources()
            LOG.info("Ray cluster resources: %s", cluster_info)
            return self.client
        except _RAY_CLIENT_EXCEPTIONS as e:
            LOG.error("Failed to connect to Ray cluster: %s", e)
            raise

    def submit_job(self, config: RayJobConfig) -> str:
        """
        Submit a job to Ray cluster.

        Args:
            config: RayJobConfig with job details

        Returns:
            Job submission ID
        """
        # Resolve the client via the connect contract: either we already have one,
        # or _connect() returns a live one (or raises). No silent None-client path.
        client = self.client or self._connect()

        # Handle dependencies: wait for prior jobs to complete
        if config.dependencies:
            self._wait_for_dependencies(config.dependencies)

        # Build runtime environment
        runtime_env = config.runtime_env or {}
        if config.env_vars:
            if "env_vars" not in runtime_env:
                runtime_env["env_vars"] = {}
            runtime_env["env_vars"].update(config.env_vars)

        # Create log directory
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)

        # Calculate per-node resources
        gpus_per_node = config.num_gpus / config.num_nodes
        cpus_per_node = config.num_cpus / config.num_nodes

        try:
            # Submit job. Ray 2.54 deprecated `job_id=`; use `submission_id=` instead.
            job_id = client.submit_job(
                entrypoint=config.command,
                submission_id=config.name,
                runtime_env=runtime_env,
                entrypoint_num_gpus=gpus_per_node,
                entrypoint_num_cpus=cpus_per_node,
            )

            LOG.info("✓ Submitted job '%s' (ID: %s)", config.name, job_id)
            LOG.info(
                "  Resources: %d node(s), %.1f GPU/node, %.1f CPU/node",
                config.num_nodes,
                gpus_per_node,
                cpus_per_node,
            )
            LOG.info("  Log dir: %s", config.log_dir)

            return job_id

        except _RAY_CLIENT_EXCEPTIONS as e:
            LOG.error("Failed to submit job %s: %s", config.name, e)
            raise

    def _wait_for_dependencies(self, job_ids: List[str], poll_interval: int = 30, timeout: int = 86400):
        """
        Wait for dependent jobs to complete.

        This is a synchronous, blocking wait — ``submit_job()`` calls this before
        submitting the next Ray job, so the calling process must stay alive across
        the entire dependency chain (hours for a multi-stage pipeline like
        SDG → SFT → eval). Distinct from Slurm ``--dependency=afterany`` which is
        fire-and-forget; a future iteration may move dependency tracking to the
        Ray task graph or an async watcher.

        Args:
            job_ids: List of job IDs to wait for
            poll_interval: How often to poll job status (seconds)
            timeout: Maximum time per dependency to wait (seconds). Each dependency
                gets its own budget; the overall wait can be up to
                ``len(job_ids) * timeout`` in the worst case.
        """
        for job_id in job_ids:
            LOG.info("Waiting for dependent job %s to complete...", job_id)
            start_time = time.time()

            while True:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Timeout waiting for job {job_id}")

                # Only swallow Ray client-layer errors from the status fetch
                # itself (network blips, transient HTTP failures wrapped as
                # RuntimeError by Ray's dashboard client). Terminal-state
                # RuntimeErrors raised below for FAILED/STOPPED propagate
                # because they are raised after the try block.
                try:
                    status = self.client.get_job_status(job_id)
                except _RAY_CLIENT_EXCEPTIONS as e:
                    LOG.debug("Transient error checking job status: %s", e)
                    time.sleep(poll_interval)
                    continue

                status_str = str(status)
                if "SUCCEEDED" in status_str:
                    LOG.info("✓ Dependent job %s completed successfully", job_id)
                    break
                if any(x in status_str for x in ["FAILED", "STOPPED"]):
                    raise RuntimeError(f"Dependent job {job_id} failed with status {status}")
                LOG.debug("Job %s status: %s", job_id, status)
                time.sleep(poll_interval)

    def get_job_status(self, job_id: str) -> str:
        """Get job status."""
        return str(self.client.get_job_status(job_id))

    def get_job_logs(self, job_id: str) -> str:
        """Get job logs (best-effort: returns "" on Ray client errors)."""
        try:
            return self.client.get_job_logs(job_id)
        except _RAY_CLIENT_EXCEPTIONS as e:
            LOG.warning("Failed to retrieve logs for job %s: %s", job_id, e)
            return ""

    def cancel_job(self, job_id: str):
        """Cancel a job (best-effort: logs a warning on Ray client errors)."""
        try:
            self.client.stop_job(job_id)
            LOG.info("✓ Cancelled job %s", job_id)
        except _RAY_CLIENT_EXCEPTIONS as e:
            LOG.warning("Failed to cancel job %s: %s", job_id, e)

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs in the cluster (best-effort: returns [] on Ray client errors)."""
        try:
            return self.client.list_jobs()
        except _RAY_CLIENT_EXCEPTIONS as e:
            LOG.error("Failed to list jobs: %s", e)
            return []


def get_ray_client(cluster_config: Dict[str, Any]) -> RayJobClient:
    """Factory function to create Ray client from cluster config."""
    ray_config = cluster_config.get("ray", {})
    return RayJobClient(ray_address=ray_config.get("address", "auto"), namespace=ray_config.get("namespace", "nemo"))


# ---------------------------------------------------------------------------
# nemo-run Executor adapter
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class RayExecutor(Executor):
    """Ray-based executor for standalone Ray clusters and Ray-on-Slurm.

    Holds Ray cluster parameters and per-job resource configuration. Actual
    job submission is performed by ``RayJobClient`` (also defined in this
    module) via the ``add_task()`` Ray routing branch.

    Example:

    .. code-block:: python

        run.RayExecutor(
            ray_address="auto",
            ray_namespace="nemo",
            num_gpus=8,
            num_cpus=64,
            num_nodes=1,
            log_dir="/workspace/logs/ray_jobs",
        )
    """

    #: Ray cluster address. Use "auto" for an existing cluster started via
    #: `ray start` (Ray-on-Slurm) or "ray://host:10001" for a remote cluster.
    ray_address: str = "auto"

    #: Ray namespace for job isolation across users/jobs on a shared cluster.
    ray_namespace: str = "nemo"

    #: Total GPUs requested across all nodes for a single job.
    num_gpus: int = 1

    #: Total CPUs requested across all nodes for a single job.
    num_cpus: int = 8

    #: Number of nodes to span. Used to derive per-node resource shares.
    num_nodes: int = 1

    #: Tasks per node — used by torchrun/launcher components for nproc derivation.
    #: For most Ray submissions (single-entrypoint), this stays at 1.
    ntasks_per_node: int = 1

    #: Directory where Ray submission metadata + logs are written. Should be on
    #: a shared filesystem visible to head + workers.
    log_dir: str = "/tmp/ray_jobs"

    def assign(
        self,
        exp_id: str,
        exp_dir: str,
        task_id: str,
        task_dir: str,
    ):
        """Set experiment-level attributes when the executor is bound to a task.

        Mirrors `LocalExecutor.assign()` to satisfy `nemo_run.Experiment` lifecycle
        expectations even though the Ray path skips `exp.add()` for actual
        submission.
        """
        self.experiment_id = exp_id
        self.experiment_dir = exp_dir
        self.job_dir = os.path.join(exp_dir, task_dir)

    def nnodes(self) -> int:
        """Return number of nodes — used by torchrun-style multi-node launchers."""
        return self.num_nodes

    def nproc_per_node(self) -> int:
        """Return processes per node — used by torchrun-style launchers.

        For Ray jobs, this is typically 1 (single entrypoint per submission).
        """
        return self.ntasks_per_node
