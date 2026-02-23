# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Slurm compute backend wrapping the existing NeMo-Run integration.

This backend provides compatibility with the new ComputeBackend interface
while delegating to the existing Slurm executor implementation in exp.py.

Current Status: TRANSITIONAL WRAPPER
------------------------------------

This is a transitional wrapper with LIMITED functionality. The existing exp.py
functions (get_executor, add_task) remain the primary interface for Slurm jobs
and should be used for production workloads.

Limitations of this wrapper:
- get_status(): Returns RUNNING as placeholder (TODO: implement via sacct)
- wait_for_completion(): Not fully implemented (TODO: poll sacct or use nemo-run)
- cancel_job(): Not implemented (TODO: implement via scancel)
- get_logs(): Not implemented (TODO: read from Slurm output files)

For full Slurm functionality (heterogeneous jobs, dependencies, code packaging),
use the existing pattern:

    from nemo_skills.pipeline.utils.exp import get_executor, add_task, get_exp

    with get_exp(expname, cluster_config) as exp:
        add_task(exp, cmd, task_name, cluster_config, ...)

TODO: Future improvements needed:
1. Implement get_status() using subprocess call to sacct
2. Implement wait_for_completion() with polling loop
3. Implement cancel_job() using scancel
4. Implement get_logs() to read from CustomJobDetails log paths
5. Consider contributing changes to nemo-run for native ComputeBackend support
"""

import logging
import shlex
from typing import Dict, Iterator, Optional

from nemo_skills.pipeline.backends.base import (
    ComputeBackend,
    JobHandle,
    JobSpec,
    JobStatus,
)
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


class SlurmBackend(ComputeBackend):
    """Slurm compute backend using NeMo-Run.

    This backend wraps the existing Slurm integration in NeMo-Skills,
    providing compatibility with the new ComputeBackend interface.

    For most use cases, the existing exp.py functions (get_executor, add_task)
    should be used directly. This wrapper is provided for scenarios where
    the unified backend interface is preferred.

    Configuration options in cluster_config:
        - executor: Must be 'slurm'
        - account: Slurm account for billing
        - partition: Default GPU partition
        - cpu_partition: CPU-only partition
        - ssh_tunnel: SSH tunnel configuration for remote clusters
        - job_dir: Directory for job metadata
        - containers: Dict mapping container types to images
        - mounts: List of volume mounts
        - env_vars: List of environment variables
        - timeouts: Dict of partition-specific timeouts
    """

    def __init__(self, cluster_config: Dict):
        self.config = cluster_config

        # Validate required fields
        if cluster_config["executor"] != "slurm":
            raise ValueError("SlurmBackend requires executor='slurm' in config")

        # Track submitted jobs
        self._jobs: Dict[str, Dict] = {}

    @property
    def name(self) -> str:
        return "slurm"

    def submit_job(self, spec: JobSpec) -> JobHandle:
        """Submit a job to Slurm.

        Note: For full Slurm functionality (heterogeneous jobs, dependencies,
        code packaging), use the existing add_task() function in exp.py.
        This method provides basic job submission for simple use cases.
        """
        # Import here to avoid circular imports
        from nemo_skills.pipeline.utils.exp import add_task, get_exp

        LOG.info(f"Submitting Slurm job: {spec.name}")

        # Convert JobSpec to add_task parameters
        # For multi-container jobs, use heterogeneous mode
        is_heterogeneous = spec.is_multi_container

        # Get the main container (or first container for multi-container)
        main_container = spec.containers[0]

        # Build command from container spec
        cmd = shlex.join(main_container.command)

        # Resolve container image
        container_image = main_container.image
        if container_image in self.config.get("containers", {}):
            container_image = self.config["containers"][container_image]

        # Submit using existing infrastructure
        with get_exp(spec.name, self.config) as exp:
            requested_gpus = main_container.resources.gpus if main_container.resources.gpus is not None else None

            task = add_task(
                exp=exp,
                cmd=cmd,
                task_name=spec.name,
                cluster_config=self.config,
                container=container_image,
                num_gpus=requested_gpus,
                num_nodes=1,
                heterogeneous=is_heterogeneous,
            )

            # Store job info
            job_info = {
                "task": task,
                "spec": spec,
                "experiment_name": spec.name,
            }
            job_id = f"slurm-{spec.name}"
            self._jobs[job_id] = job_info

        return JobHandle(
            job_id=job_id,
            backend="slurm",
            metadata={"experiment_name": spec.name},
        )

    def get_status(self, handle: JobHandle) -> JobStatus:
        """Get job status from Slurm.

        Note: Full status tracking requires integration with NeMo-Run's
        experiment tracking. This is a simplified implementation.
        """
        job_info = self._jobs.get(handle.job_id)
        if job_info is None:
            return JobStatus.UNKNOWN

        # TODO: Query actual Slurm status via sacct or NeMo-Run
        # For now, return RUNNING as a placeholder
        LOG.warning("SlurmBackend.get_status() is not fully implemented")
        return JobStatus.RUNNING

    def wait_for_completion(self, handle: JobHandle, timeout: Optional[int] = None) -> JobStatus:
        """Wait for Slurm job to complete.

        Note: This requires integration with NeMo-Run's experiment tracking.
        """
        LOG.warning("SlurmBackend.wait_for_completion() is not fully implemented")
        # TODO: Implement using NeMo-Run's wait functionality
        return self.get_status(handle)

    def cancel_job(self, handle: JobHandle) -> bool:
        """Cancel a Slurm job."""
        job_info = self._jobs.get(handle.job_id)
        if job_info is None:
            return False

        LOG.warning("SlurmBackend.cancel_job() is not fully implemented")
        # TODO: Implement using scancel or NeMo-Run
        return False

    def get_logs(
        self,
        handle: JobHandle,
        container: Optional[str] = None,
        follow: bool = False,
    ) -> Iterator[str]:
        """Get logs from a Slurm job.

        Slurm logs are typically written to files. This method would need
        to read from the log files specified in CustomJobDetails.
        """
        LOG.warning("SlurmBackend.get_logs() is not fully implemented")
        # TODO: Read from Slurm log files
        yield f"[SlurmBackend] Logs for job {handle.job_id} not yet implemented"

    def cleanup(self, handle: JobHandle) -> None:
        """Clean up job resources."""
        self._jobs.pop(handle.job_id, None)

    def get_internal_address(self, port: int) -> str:
        """Get address for inter-container communication in Slurm.

        For heterogeneous Slurm jobs, containers communicate via
        the master node hostname.
        """
        # In Slurm het jobs, use SLURM_MASTER_NODE environment variable
        return f"$SLURM_MASTER_NODE:{port}"

    def health_check(self) -> bool:
        """Check if Slurm is accessible.

        For remote clusters, this checks SSH connectivity.
        For local clusters, this checks if squeue is available.
        """
        import subprocess

        ssh_tunnel = self.config.get("ssh_tunnel")

        if ssh_tunnel:
            # Check SSH connectivity
            try:
                result = subprocess.run(
                    [
                        "ssh",
                        "-o",
                        "ConnectTimeout=5",
                        "-o",
                        "BatchMode=yes",
                        f"{ssh_tunnel['user']}@{ssh_tunnel['host']}",
                        "echo",
                        "ok",
                    ],
                    capture_output=True,
                    timeout=10,
                )
                return result.returncode == 0 and b"ok" in result.stdout
            except Exception as e:
                LOG.warning(f"SSH health check failed: {e}")
                return False
        else:
            # Check local squeue
            try:
                result = subprocess.run(
                    ["squeue", "--version"],
                    capture_output=True,
                    timeout=5,
                )
                return result.returncode == 0
            except Exception as e:
                LOG.warning(f"Slurm health check failed: {e}")
                return False
