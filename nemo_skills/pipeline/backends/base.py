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

"""Base classes and interfaces for compute backends.

This module defines the abstract interface that all compute backends must implement,
along with the data classes used to specify jobs and their resources.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional


class JobStatus(Enum):
    """Status of a submitted job."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


@dataclass
class ResourceSpec:
    """Resource requirements for a container.

    Attributes:
        gpus: Number of GPUs required (default: 0).
        cpus: Number of CPU cores required (default: 1).
        memory_gb: Memory in gigabytes (default: 4.0).
    """

    gpus: int = 0
    cpus: int = 1
    memory_gb: float = 4.0

    def __post_init__(self):
        if self.gpus < 0:
            raise ValueError(f"gpus must be non-negative, got {self.gpus}")
        if self.cpus < 1:
            raise ValueError(f"cpus must be at least 1, got {self.cpus}")
        if self.memory_gb <= 0:
            raise ValueError(f"memory_gb must be positive, got {self.memory_gb}")


@dataclass
class ContainerSpec:
    """Specification for a container within a job.

    Attributes:
        name: Unique name for the container within the job.
        image: Container image to use (e.g., 'nvcr.io/nvidia/nemo:latest').
        command: Command to run in the container.
        env_vars: Environment variables to set.
        mounts: Volume mounts in 'src:dst[:ro]' format.
        resources: Resource requirements for this container.
        ports: Ports to expose from this container.
        working_dir: Working directory inside the container.
    """

    name: str
    image: str
    command: List[str]
    env_vars: Dict[str, str] = field(default_factory=dict)
    mounts: List[str] = field(default_factory=list)
    resources: ResourceSpec = field(default_factory=ResourceSpec)
    ports: List[int] = field(default_factory=list)
    working_dir: Optional[str] = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("Container name cannot be empty")
        if not self.image:
            raise ValueError("Container image cannot be empty")


@dataclass
class JobSpec:
    """Specification for a job to be submitted.

    A job can contain one or more containers. For single-container jobs,
    provide a list with one ContainerSpec. For multi-container jobs
    (e.g., server + client pattern), provide multiple ContainerSpecs.

    Multi-container jobs run in the same allocation and can communicate
    via localhost.

    Attributes:
        name: Unique name for the job.
        containers: List of containers to run (single = simple job, multiple = colocated).
        timeout_seconds: Maximum runtime in seconds (None = no limit).
        dependencies: Job IDs that must complete before this job starts.
        labels: Key-value labels for the job.
        node_selector: Node selection criteria (maps to K8s nodeSelector or Slurm partition).
        annotations: Additional metadata annotations.
    """

    name: str
    containers: List[ContainerSpec]
    timeout_seconds: Optional[int] = None
    dependencies: Optional[List[str]] = None
    labels: Optional[Dict[str, str]] = None
    node_selector: Optional[Dict[str, str]] = None
    annotations: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("Job name cannot be empty")
        if not self.containers:
            raise ValueError("Job must have at least one container")
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}")

    @property
    def is_multi_container(self) -> bool:
        """Check if this is a multi-container (colocated) job."""
        return len(self.containers) > 1

    @property
    def total_gpus(self) -> int:
        """Total GPUs across all containers."""
        return sum(c.resources.gpus for c in self.containers)


@dataclass
class JobHandle:
    """Handle for tracking a submitted job.

    Attributes:
        job_id: Unique identifier for the job (backend-specific).
        backend: Name of the backend that created this handle.
        metadata: Backend-specific metadata for job management.
    """

    job_id: str
    backend: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"JobHandle({self.backend}:{self.job_id})"


class ComputeBackend(ABC):
    """Abstract interface for compute backends.

    All compute backends (Slurm, Kubernetes, Local) must implement this interface
    to provide a unified way to submit and manage jobs.

    Example usage:
        backend = BackendFactory.get_backend(cluster_config)
        handle = backend.submit_job(job_spec)
        status = backend.wait_for_completion(handle)
        for line in backend.get_logs(handle):
            print(line)
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this backend (e.g., 'slurm', 'kubernetes', 'local')."""
        pass

    @abstractmethod
    def submit_job(self, spec: JobSpec) -> JobHandle:
        """Submit a job and return a handle for tracking.

        For multi-container jobs, all containers are scheduled together
        and can communicate via localhost.

        Args:
            spec: Job specification including containers and resources.

        Returns:
            JobHandle for tracking the submitted job.

        Raises:
            RuntimeError: If job submission fails.
        """
        pass

    @abstractmethod
    def get_status(self, handle: JobHandle) -> JobStatus:
        """Get the current status of a job.

        Args:
            handle: Handle returned from submit_job().

        Returns:
            Current JobStatus.
        """
        pass

    @abstractmethod
    def wait_for_completion(
        self, handle: JobHandle, timeout: Optional[int] = None
    ) -> JobStatus:
        """Block until the job completes or timeout is reached.

        Args:
            handle: Handle returned from submit_job().
            timeout: Maximum seconds to wait (None = wait indefinitely).

        Returns:
            Final JobStatus (SUCCEEDED, FAILED, or CANCELLED).
            Returns current status if timeout is reached.
        """
        pass

    @abstractmethod
    def cancel_job(self, handle: JobHandle) -> bool:
        """Cancel a running or pending job.

        Args:
            handle: Handle returned from submit_job().

        Returns:
            True if cancellation was successful, False otherwise.
        """
        pass

    @abstractmethod
    def get_logs(
        self,
        handle: JobHandle,
        container: Optional[str] = None,
        follow: bool = False,
    ) -> Iterator[str]:
        """Stream logs from a job.

        Args:
            handle: Handle returned from submit_job().
            container: Specific container name (for multi-container jobs).
                      If None, returns logs from the first/main container.
            follow: If True, continue streaming as new logs arrive.

        Yields:
            Log lines as strings.
        """
        pass

    @abstractmethod
    def cleanup(self, handle: JobHandle) -> None:
        """Clean up resources associated with a job.

        This should be called after job completion to release resources.
        Implementations should be idempotent (safe to call multiple times).

        Args:
            handle: Handle returned from submit_job().
        """
        pass

    def get_internal_address(self, port: int) -> str:
        """Get address for inter-container communication within a job.

        For colocated containers (multi-container pod/heterogeneous job),
        this returns localhost since containers share a network namespace.

        Override this method for backends that use different addressing
        (e.g., distributed multi-node jobs).

        Args:
            port: Port number for the service.

        Returns:
            Address string (e.g., 'localhost:8000').
        """
        return f"localhost:{port}"

    def health_check(self) -> bool:
        """Check if the backend is available and properly configured.

        Returns:
            True if the backend is healthy, False otherwise.
        """
        return True
