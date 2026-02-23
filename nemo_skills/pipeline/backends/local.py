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

"""Local compute backend for running jobs locally or in Docker.

This backend wraps the existing LocalExecutor and DockerExecutor from NeMo-Run
to provide compatibility with the new ComputeBackend interface.
"""

import logging
import subprocess
import threading
import time
import uuid
from typing import Dict, Iterator, List, Optional

from nemo_skills.pipeline.backends.base import (
    ComputeBackend,
    ContainerSpec,
    JobHandle,
    JobSpec,
    JobStatus,
)
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


class LocalJob:
    """Tracks a locally running job."""

    def __init__(self, job_id: str, processes: List[subprocess.Popen], spec: JobSpec):
        self.job_id = job_id
        self.processes = processes
        self.spec = spec
        self.start_time = time.time()
        self.logs: Dict[str, List[str]] = {c.name: [] for c in spec.containers}
        self._log_threads: List[threading.Thread] = []
        self._cancelled = False

    @property
    def status(self) -> JobStatus:
        if self._cancelled:
            return JobStatus.CANCELLED

        # Check if all processes have completed
        all_done = all(p.poll() is not None for p in self.processes)

        if not all_done:
            return JobStatus.RUNNING

        # All done - check return codes
        return_codes = [p.returncode for p in self.processes]
        if all(rc == 0 for rc in return_codes):
            return JobStatus.SUCCEEDED
        else:
            return JobStatus.FAILED

    def cancel(self):
        """Cancel all processes."""
        self._cancelled = True
        for p in self.processes:
            if p.poll() is None:
                p.terminate()
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    p.kill()


class LocalBackend(ComputeBackend):
    """Local compute backend for development and testing.

    This backend runs jobs locally, either directly or via Docker containers.
    It supports multi-container jobs by running containers in parallel.

    Configuration options in cluster_config:
        - executor: 'local' (Docker) or 'none' (direct execution)
        - containers: Dict mapping container names to images
        - mounts: List of volume mounts
        - env_vars: List of environment variables
    """

    def __init__(self, cluster_config: Dict):
        self.config = cluster_config
        self.use_docker = cluster_config.get("executor") == "local"
        self._jobs: Dict[str, LocalJob] = {}

    @property
    def name(self) -> str:
        return "local" if self.use_docker else "none"

    def submit_job(self, spec: JobSpec) -> JobHandle:
        """Submit a job for local execution."""
        job_id = f"local-{spec.name}-{uuid.uuid4().hex[:8]}"

        processes = []
        for container_spec in spec.containers:
            if self.use_docker:
                cmd = self._build_docker_command(container_spec, spec)
            else:
                cmd = container_spec.command

            LOG.info(f"Starting container {container_spec.name}: {' '.join(cmd)}")

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            processes.append(proc)

        job = LocalJob(job_id, processes, spec)
        self._jobs[job_id] = job

        # Start log collection threads
        for i, (proc, container_spec) in enumerate(zip(processes, spec.containers)):
            thread = threading.Thread(
                target=self._collect_logs,
                args=(job, container_spec.name, proc),
                daemon=True,
            )
            thread.start()
            job._log_threads.append(thread)

        return JobHandle(
            job_id=job_id,
            backend=self.name,
            metadata={"spec": spec},
        )

    def _build_docker_command(self, container_spec: ContainerSpec, job_spec: JobSpec) -> List[str]:
        """Build docker run command."""
        cmd = ["docker", "run", "--rm"]

        # Add name
        cmd.extend(["--name", f"{job_spec.name}-{container_spec.name}"])

        # Add network mode (host for localhost communication)
        cmd.extend(["--network", "host"])

        # Add GPU support
        if container_spec.resources.gpus > 0:
            cmd.extend(["--gpus", str(container_spec.resources.gpus)])

        # Add environment variables
        for key, value in container_spec.env_vars.items():
            cmd.extend(["-e", f"{key}={value}"])

        # Add mounts
        for mount in container_spec.mounts:
            cmd.extend(["-v", mount])

        # Add mounts from config
        for mount in self.config.get("mounts", []):
            cmd.extend(["-v", mount])

        # Add working directory
        if container_spec.working_dir:
            cmd.extend(["-w", container_spec.working_dir])

        # Add image
        cmd.append(container_spec.image)

        # Add command
        cmd.extend(container_spec.command)

        return cmd

    def _collect_logs(self, job: LocalJob, container_name: str, proc: subprocess.Popen):
        """Collect logs from a process."""
        try:
            for line in proc.stdout:
                job.logs[container_name].append(line)
        except Exception as e:
            LOG.warning(f"Error collecting logs for {container_name}: {e}")

    def get_status(self, handle: JobHandle) -> JobStatus:
        """Get job status."""
        job = self._jobs.get(handle.job_id)
        if job is None:
            return JobStatus.UNKNOWN
        return job.status

    def wait_for_completion(self, handle: JobHandle, timeout: Optional[int] = None) -> JobStatus:
        """Wait for job to complete."""
        job = self._jobs.get(handle.job_id)
        if job is None:
            return JobStatus.UNKNOWN

        start_time = time.time()
        while True:
            status = job.status
            if status in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED):
                return status

            if timeout is not None and (time.time() - start_time) > timeout:
                return status

            time.sleep(1)

    def cancel_job(self, handle: JobHandle) -> bool:
        """Cancel a running job."""
        job = self._jobs.get(handle.job_id)
        if job is None:
            return False

        job.cancel()
        return True

    def get_logs(
        self,
        handle: JobHandle,
        container: Optional[str] = None,
        follow: bool = False,
    ) -> Iterator[str]:
        """Get logs from a job."""
        job = self._jobs.get(handle.job_id)
        if job is None:
            return

        if container is None:
            container = job.spec.containers[0].name

        logs = job.logs.get(container, [])

        # Yield existing logs
        yielded = 0
        for line in logs:
            yield line
            yielded += 1

        # If following, continue yielding new logs
        if follow:
            while job.status == JobStatus.RUNNING:
                current_logs = job.logs.get(container, [])
                while yielded < len(current_logs):
                    yield current_logs[yielded]
                    yielded += 1
                time.sleep(0.1)

            # Yield any remaining logs
            current_logs = job.logs.get(container, [])
            while yielded < len(current_logs):
                yield current_logs[yielded]
                yielded += 1

    def cleanup(self, handle: JobHandle) -> None:
        """Clean up job resources."""
        job = self._jobs.pop(handle.job_id, None)
        if job:
            job.cancel()

    def health_check(self) -> bool:
        """Check if Docker is available (if using Docker mode)."""
        if not self.use_docker:
            return True

        try:
            result = subprocess.run(
                ["docker", "info"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except Exception as e:
            LOG.warning(f"Docker health check failed: {e}")
            return False
