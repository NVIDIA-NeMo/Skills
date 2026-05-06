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

"""Ray cluster job submission client for distributed training without SLURM."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import ray
    from ray.job_submission import JobSubmissionClient
except ImportError:
    raise ImportError("ray is required for Ray executor. Install with: pip install ray")

LOG = logging.getLogger(__name__)


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
        """Connect to Ray cluster."""
        try:
            if not ray.is_initialized():
                ray.init(address=self.ray_address, namespace=self.namespace,
                         ignore_reinit_error=True)
            self.client = JobSubmissionClient(address=self.ray_address)
            LOG.info(f"Connected to Ray cluster at {self.ray_address}")

            # Get cluster info
            cluster_info = ray.cluster_resources()
            LOG.info(f"Ray cluster resources: {cluster_info}")
        except Exception as e:
            LOG.error(f"Failed to connect to Ray cluster: {e}")
            raise

    def submit_job(self, config: RayJobConfig) -> str:
        """
        Submit a job to Ray cluster.

        Args:
            config: RayJobConfig with job details

        Returns:
            Job submission ID
        """
        if not self.client:
            self._connect()

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
            job_id = self.client.submit_job(
                entrypoint=config.command,
                submission_id=config.name,
                runtime_env=runtime_env,
                entrypoint_num_gpus=gpus_per_node,
                entrypoint_num_cpus=cpus_per_node,
            )

            LOG.info(f"✓ Submitted job '{config.name}' (ID: {job_id})")
            LOG.info(f"  Resources: {config.num_nodes} node(s), {gpus_per_node:.1f} GPU/node, "
                     f"{cpus_per_node:.1f} CPU/node")
            LOG.info(f"  Log dir: {config.log_dir}")

            return job_id

        except Exception as e:
            LOG.error(f"Failed to submit job {config.name}: {e}")
            raise

    def _wait_for_dependencies(self, job_ids: List[str], poll_interval: int = 30, timeout: int = 86400):
        """
        Wait for dependent jobs to complete.

        Args:
            job_ids: List of job IDs to wait for
            poll_interval: How often to poll job status (seconds)
            timeout: Maximum time to wait (seconds)
        """
        start_time = time.time()

        for job_id in job_ids:
            LOG.info(f"Waiting for dependent job {job_id} to complete...")

            while True:
                if time.time() - start_time > timeout:
                    raise TimeoutError(f"Timeout waiting for job {job_id}")

                try:
                    status = self.client.get_job_status(job_id)
                    status_str = str(status)

                    if "SUCCEEDED" in status_str:
                        LOG.info(f"✓ Dependent job {job_id} completed successfully")
                        break
                    elif any(x in status_str for x in ["FAILED", "STOPPED"]):
                        raise RuntimeError(f"Dependent job {job_id} failed with status {status}")
                    else:
                        LOG.debug(f"Job {job_id} status: {status}")
                        time.sleep(poll_interval)

                except Exception as e:
                    LOG.debug(f"Error checking job status: {e}")
                    time.sleep(poll_interval)

    def get_job_status(self, job_id: str) -> str:
        """Get job status."""
        return str(self.client.get_job_status(job_id))

    def get_job_logs(self, job_id: str) -> str:
        """Get job logs."""
        try:
            return self.client.get_job_logs(job_id)
        except Exception as e:
            LOG.warning(f"Failed to retrieve logs for job {job_id}: {e}")
            return ""

    def cancel_job(self, job_id: str):
        """Cancel a job."""
        try:
            self.client.stop_job(job_id)
            LOG.info(f"✓ Cancelled job {job_id}")
        except Exception as e:
            LOG.warning(f"Failed to cancel job {job_id}: {e}")

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs in the cluster."""
        try:
            return self.client.list_jobs()
        except Exception as e:
            LOG.error(f"Failed to list jobs: {e}")
            return []


def get_ray_client(cluster_config: Dict[str, Any]) -> RayJobClient:
    """Factory function to create Ray client from cluster config."""
    ray_config = cluster_config.get("ray", {})
    return RayJobClient(
        ray_address=ray_config.get("address", "auto"),
        namespace=ray_config.get("namespace", "nemo")
    )
