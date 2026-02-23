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

"""Integration module for compute backends with existing NeMo-Skills infrastructure.

Architecture Overview
---------------------

NeMo-Skills has two parallel job submission interfaces:

1. **Legacy Interface (nemo-run based)**:
   - Entry point: `get_executor()` + `add_task()` in `nemo_skills/pipeline/utils/exp.py`
   - Returns nemo-run executor objects (LocalExecutor, DockerExecutor, SlurmExecutor)
   - Tightly coupled to nemo-run's abstractions and code packaging
   - Full feature support for Slurm (heterogeneous jobs, dependencies, code upload)

2. **New Interface (ComputeBackend based)**:
   - Entry point: `get_backend()` in `nemo_skills/pipeline/backends/factory.py`
   - Returns ComputeBackend instances (LocalBackend, KubernetesBackend, SlurmBackend)
   - Clean abstraction layer supporting multiple backends uniformly
   - Native Kubernetes support with multi-container Pods

Bridge Strategy
---------------

This module serves as a bridge between the two interfaces. The recommended
approach depends on your target backend:

**For Kubernetes workloads** - Use the new backend interface directly:

    from nemo_skills.pipeline.backends import get_backend, JobSpec, ContainerSpec

    backend = get_backend(cluster_config)
    spec = JobSpec(name="my-job", containers=[...])
    handle = backend.submit_job(spec)
    status = backend.wait_for_completion(handle)

**For Slurm/local workloads** - The existing nemo-run pattern is recommended
for full feature support (code packaging, heterogeneous jobs, dependencies):

    from nemo_skills.pipeline.utils.exp import get_executor, add_task, get_exp

    with get_exp(expname, cluster_config) as exp:
        add_task(exp, cmd, task_name, cluster_config, ...)

TODO: NeMo-Run Integration
--------------------------

The current implementation has the following limitations that require future work:

1. **No KubernetesExecutor in nemo-run**: The nemo-run library doesn't have a
   native Kubernetes executor. Options for full integration:

   a. Contribute a KubernetesExecutor to nemo-run that wraps our KubernetesBackend
   b. Create an adapter class that implements nemo-run's executor interface
   c. Modify get_executor() to return a compatible wrapper for K8s

2. **Code packaging for K8s**: nemo-run's Packager handles code upload for remote
   execution on Slurm. For Kubernetes, we need:

   - Container image builds with code baked in, OR
   - PVC-based code mounting, OR
   - Init containers that fetch code from git/S3

3. **SlurmBackend is a thin wrapper**: The current SlurmBackend delegates to
   the existing add_task() infrastructure. A proper implementation would:

   - Implement full job lifecycle management
   - Query job status via sacct
   - Support job cancellation via scancel
   - Handle log retrieval from Slurm output files

4. **Unified CLI entry point**: Pipeline scripts (generate.py, eval.py, etc.)
   currently use get_executor(). To support K8s transparently:

   - Add executor='kubernetes' detection in pipeline scripts
   - Route to appropriate backend based on config
   - Maintain backward compatibility with existing Slurm workflows

For new Kubernetes-first workflows, use this module's helpers directly.
For migrating existing Slurm workflows, the legacy interface remains stable.
"""

import logging
from typing import Dict, List, Optional

from nemo_skills.pipeline.backends.base import (
    ContainerSpec,
    JobHandle,
    JobSpec,
    JobStatus,
    ResourceSpec,
)
from nemo_skills.pipeline.backends.factory import get_backend
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def create_inference_job_spec(
    job_name: str,
    server_image: str,
    client_image: str,
    server_command: List[str],
    client_command: List[str],
    server_gpus: int = 8,
    server_memory_gb: int = 64,
    server_port: int = 8000,
    client_env: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
    timeout_seconds: Optional[int] = None,
) -> JobSpec:
    """Create a JobSpec for the common server+client inference pattern.

    This helper creates a multi-container job specification for the typical
    NeMo-Skills inference pattern where a model server (e.g., vLLM, TensorRT-LLM)
    runs alongside a client that sends requests.

    In Kubernetes, both containers run in the same Pod and communicate via localhost.
    In Slurm heterogeneous jobs, they communicate via the master node.

    Args:
        job_name: Name for the job.
        server_image: Container image for the inference server.
        client_image: Container image for the client.
        server_command: Command to start the server.
        client_command: Command to run the client.
        server_gpus: Number of GPUs for the server (default: 8).
        server_memory_gb: Memory allocation for server in GB (default: 64).
        server_port: Port the server listens on (default: 8000).
        client_env: Additional environment variables for the client.
        labels: Labels to apply to the job.
        timeout_seconds: Job timeout in seconds.

    Returns:
        JobSpec configured for server+client inference.

    Example:
        spec = create_inference_job_spec(
            job_name="llama-inference",
            server_image="vllm",
            client_image="nemo-skills",
            server_command=["python", "-m", "vllm.entrypoints.api_server",
                          "--model", "/models/llama-70b"],
            client_command=["python", "generate.py", "--server", "localhost:8000"],
            server_gpus=8,
        )
    """
    server = ContainerSpec(
        name="server",
        image=server_image,
        command=server_command,
        resources=ResourceSpec(gpus=server_gpus, memory_gb=server_memory_gb),
        ports=[server_port],
    )

    env = {"SERVER_ADDRESS": f"localhost:{server_port}"}
    if client_env:
        env.update(client_env)

    client = ContainerSpec(
        name="client",
        image=client_image,
        command=client_command,
        resources=ResourceSpec(gpus=0, memory_gb=16),
        env_vars=env,
    )

    return JobSpec(
        name=job_name,
        containers=[server, client],
        labels=labels,
        timeout_seconds=timeout_seconds,
    )


def create_training_job_spec(
    job_name: str,
    image: str,
    command: List[str],
    gpus: int = 8,
    memory_gb: int = 128,
    env_vars: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
    timeout_seconds: Optional[int] = None,
) -> JobSpec:
    """Create a JobSpec for a training job.

    Args:
        job_name: Name for the job.
        image: Container image for training.
        command: Training command.
        gpus: Number of GPUs (default: 8).
        memory_gb: Memory allocation in GB (default: 128).
        env_vars: Environment variables.
        labels: Labels to apply to the job.
        timeout_seconds: Job timeout in seconds.

    Returns:
        JobSpec configured for training.
    """
    container = ContainerSpec(
        name="trainer",
        image=image,
        command=command,
        resources=ResourceSpec(gpus=gpus, memory_gb=memory_gb),
        env_vars=env_vars or {},
    )

    return JobSpec(
        name=job_name,
        containers=[container],
        labels=labels,
        timeout_seconds=timeout_seconds,
    )


def create_data_processing_job_spec(
    job_name: str,
    image: str,
    command: List[str],
    cpus: int = 16,
    memory_gb: int = 64,
    env_vars: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, str]] = None,
    timeout_seconds: Optional[int] = None,
) -> JobSpec:
    """Create a JobSpec for a CPU-only data processing job.

    Args:
        job_name: Name for the job.
        image: Container image.
        command: Processing command.
        cpus: Number of CPUs (default: 16).
        memory_gb: Memory allocation in GB (default: 64).
        env_vars: Environment variables.
        labels: Labels to apply to the job.
        timeout_seconds: Job timeout in seconds.

    Returns:
        JobSpec configured for data processing.
    """
    container = ContainerSpec(
        name="processor",
        image=image,
        command=command,
        resources=ResourceSpec(gpus=0, cpus=cpus, memory_gb=memory_gb),
        env_vars=env_vars or {},
    )

    return JobSpec(
        name=job_name,
        containers=[container],
        labels=labels,
        timeout_seconds=timeout_seconds,
    )


def run_job_and_wait(
    cluster_config: Dict,
    spec: JobSpec,
    timeout: Optional[int] = None,
) -> JobStatus:
    """Submit a job and wait for completion.

    Convenience function that combines get_backend, submit_job, and
    wait_for_completion into a single call.

    Args:
        cluster_config: Cluster configuration dict.
        spec: Job specification.
        timeout: Maximum time to wait in seconds.

    Returns:
        Final job status.

    Example:
        status = run_job_and_wait(
            cluster_config,
            create_training_job_spec("sft-job", "nemo-skills", ["python", "train.py"]),
        )
        if status == JobStatus.SUCCEEDED:
            print("Training complete!")
    """
    backend = get_backend(cluster_config)
    handle = backend.submit_job(spec)
    LOG.info(f"Submitted job {handle.job_id} to {backend.name} backend")

    status = backend.wait_for_completion(handle, timeout=timeout)
    LOG.info(f"Job {handle.job_id} finished with status: {status.value}")

    return status


def is_kubernetes_cluster(cluster_config: Dict) -> bool:
    """Check if the cluster config specifies Kubernetes.

    Args:
        cluster_config: Cluster configuration dict.

    Returns:
        True if executor is 'kubernetes'.
    """
    return cluster_config.get("executor") == "kubernetes"


def is_slurm_cluster(cluster_config: Dict) -> bool:
    """Check if the cluster config specifies Slurm.

    Args:
        cluster_config: Cluster configuration dict.

    Returns:
        True if executor is 'slurm'.
    """
    return cluster_config.get("executor") == "slurm"


def is_local_executor(cluster_config: Dict) -> bool:
    """Check if the cluster config specifies local execution.

    Args:
        cluster_config: Cluster configuration dict.

    Returns:
        True if executor is 'local' or 'none'.
    """
    return cluster_config.get("executor") in ("local", "none")
