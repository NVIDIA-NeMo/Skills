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

"""RayExecutor — staged Ray support for NeMo-Skills.

Provides a top-level Ray scheduler executor for standalone Ray clusters and
Ray-on-Slurm. Distinct from `nemo_run.core.execution.kuberay.KubeRayExecutor`,
which targets Kubernetes-managed Ray clusters via the kubernetes client API.

This RayExecutor is a configuration dataclass that holds Ray-specific cluster
parameters; actual job submission is performed by `RayJobClient` (see
`ray_executor_client.py`). Routing happens in `add_task()` / `Pipeline.run()`:
when the executor is a `RayExecutor` instance, the caller submits via
`RayJobClient.submit_job()` instead of `nemo_run.Experiment.add()`.

Supported in this MR:
- Simple single-command Ray jobs (SFT, eventually GRPO)
- Dependency chaining by Ray submission IDs
- Shared-FS runtime/code visibility

Out of scope (raise NotImplementedError at the routing layer):
- Sandbox judge containers
- Server co-scheduling (vLLM/SGLang lifecycle alongside main job)
- Heterogeneous tasks
- Multi-command task groups
- Generic eval/generate/server orchestration
"""

from dataclasses import dataclass

from nemo_run.core.execution.base import Executor


@dataclass(kw_only=True)
class RayExecutor(Executor):
    """Ray-based executor for standalone Ray clusters and Ray-on-Slurm.

    Holds Ray cluster parameters and per-job resource configuration. Actual job
    submission is performed by `RayJobClient` (see `ray_executor_client.py`)
    via the `add_task()` Ray routing branch.

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
        import os
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
