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

"""
Simplified declarative pipeline system using only Command for all task types.

Basic Example (Single job with multiple commands):
    from nemo_skills.pipeline.utils.commands import vllm_server_command, sandbox_command
    from nemo_skills.pipeline.utils.declarative import Command, CommandGroup, HardwareConfig, Pipeline
    from nemo_skills.pipeline.utils.server import get_free_port

    # Allocate ports for server and sandbox
    server_port = get_free_port(strategy="random")
    sandbox_port = get_free_port(strategy="random")

    # Commands that run together in one SLURM job
    # Note: Lambdas are needed for cross-component references (hostname_ref, meta_ref)
    # which aren't resolved until het_group_index is assigned at pipeline execution time.
    server_cmd, server_meta = vllm_server_command(cluster_cfg, model="Qwen/Qwen3-8B", port=server_port)
    server = Command(command=server_cmd, gpus=8, name="server", metadata=server_meta)

    sandbox_cmd, sandbox_meta = sandbox_command(cluster_cfg, port=sandbox_port)
    sandbox = Command(command=sandbox_cmd, name="sandbox", metadata=sandbox_meta)

    # This lambda is ESSENTIAL - server.hostname_ref() and meta_ref() aren't available until runtime
    # Client needs NEMO_SKILLS_SANDBOX_PORT to connect to sandbox
    client = Command(
        command=lambda: f"curl {server.hostname_ref()}:{server.meta_ref('port')}/health",
        name="client",
        metadata={"environment": {"NEMO_SKILLS_SANDBOX_PORT": str(sandbox_port)}}
    )

    # Group them together
    inference_group = CommandGroup(
        commands=[server, sandbox, client],
        hardware=HardwareConfig(partition="batch"),
        name="inference"
    )

    # Create and run pipeline
    pipeline = Pipeline(
        name="my_inference",
        cluster_config=cluster_config,
        jobs=[{"name": "inference", "group": inference_group}]
    )
    pipeline.run()

Advanced Example (Multiple jobs with dependencies and heterogeneous components):
    log_dir = "/experiments/full_pipeline/logs"
    # Job 1: Preprocessing
    preprocess = Command(
        command="python preprocess.py --input data.jsonl --output processed.jsonl",
        gpus=0,
        name="preprocess"
    )
    prep_group = CommandGroup(
        commands=[preprocess],
        hardware=HardwareConfig(partition="cpu"),
        name="prep",
        log_dir=log_dir
    )
    prep_job = {"name": "prep", "group": prep_group}

    # Job 2: Two different model servers (HETEROGENEOUS SLURM job with 2 het components)
    # Allocate ports for each server/sandbox pair
    from nemo_skills.pipeline.utils.server import get_free_port
    server_8b_port = get_free_port(strategy="random")
    sandbox_8b_port = get_free_port(strategy="random")
    server_32b_port = get_free_port(strategy="random")
    sandbox_32b_port = get_free_port(strategy="random")

    # Build commands with cluster_config
    server_8b_cmd, server_8b_meta = vllm_server_command(cluster_config, model="Qwen/Qwen3-8B", port=server_8b_port)
    sandbox_8b_cmd, sandbox_8b_meta = sandbox_command(cluster_config, port=sandbox_8b_port)
    server_32b_cmd, server_32b_meta = vllm_server_command(cluster_config, model="Qwen/Qwen3-32B", port=server_32b_port)
    sandbox_32b_cmd, sandbox_32b_meta = sandbox_command(cluster_config, port=sandbox_32b_port)

    server_8b = Command(command=server_8b_cmd, gpus=8, name="server_8b", metadata=server_8b_meta)
    sandbox_8b = Command(command=sandbox_8b_cmd, name="sandbox_8b", metadata=sandbox_8b_meta)
    eval_8b = Command(command="python eval.py --model 8b", gpus=1, name="eval_8b")

    server_32b = Command(command=server_32b_cmd, gpus=8, name="server_32b", metadata=server_32b_meta)
    sandbox_32b = Command(command=sandbox_32b_cmd, name="sandbox_32b", metadata=sandbox_32b_meta)
    eval_32b = Command(command="python eval.py --model 32b", gpus=1, name="eval_32b")

    group_8b = CommandGroup(commands=[server_8b, sandbox_8b, eval_8b], name="eval_8b", log_dir=log_dir)
    group_32b = CommandGroup(commands=[server_32b, sandbox_32b, eval_32b], name="eval_32b", log_dir=log_dir)

    evals_job = {"name": "evals", "groups": [group_8b, group_32b], "dependencies": [prep_job]}

    # Job 3: Report generation (depends on both evaluations)
    report = Command(
        command="python generate_report.py --output report.txt",
        gpus=0,
        name="report"
    )
    report_group = CommandGroup(commands=[report], name="report", log_dir=log_dir)

    # Create pipeline with dependency graph
    pipeline = Pipeline(
        name="full_pipeline",
        cluster_config=cluster_config,
        jobs=[
            prep_job,
            evals_job,
            # Report depends on the eval job (internal) and some external experiment (string)
            {"name": "report", "group": report_group, "dependencies": [evals_job, "external_training_exp"]},
        ]
    )
    pipeline.run()
"""

import logging
import shlex
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import nemo_run as run

from nemo_skills.pipeline.utils import (
    get_env_variables,
    get_executor,
    get_exp,
    get_exp_handles,
    get_tunnel,
    run_exp,
    temporary_env_update,
)
from nemo_skills.pipeline.utils.commands import wrap_command
from nemo_skills.pipeline.utils.exp import REUSE_CODE_EXP, get_packaging_job_key, install_packages_wrap, tunnel_hash
from nemo_skills.pipeline.utils.mounts import is_mounted_filepath
from nemo_skills.pipeline.utils.packager import get_registered_external_repo
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


@dataclass
class Command:
    """Declarative command for running tasks in containers.

    The command can be either:
    - A string: evaluated immediately
    - A callable (lambda): evaluated lazily when the task is prepared

    Lambdas are ONLY needed for cross-component references (hostname_ref, meta_ref).
    The het_group_index isn't assigned until pipeline execution, so these must be lazy:
        # Lambda is ESSENTIAL here - server.hostname_ref() and meta_ref() don't exist yet
        client = Command(command=lambda: f"curl {server.hostname_ref()}:{server.meta_ref('port')}")
    """

    # Command can be a string or callable (lambda).
    # Lambdas are primarily used for cross-component references (hostname_ref, meta_ref).
    command: Union[str, Callable]
    container: str = "nemo-skills"
    gpus: Optional[int] = None
    nodes: int = 1
    name: str = "command"
    working_dir: str = "/nemo_run/code"
    env_vars: Dict[str, str] = field(default_factory=dict)
    installation_command: Optional[str] = None
    port: Optional[int] = None  # Can be set from metadata
    metadata: Dict[str, any] = field(default_factory=dict)  # Stores metadata from command builders
    het_group_index: Optional[int] = None  # Set per-job by Pipeline (not global)

    def __post_init__(self):
        # Wrap plain strings with environment setup
        if isinstance(self.command, str) and (self.env_vars or self.working_dir):
            self.command = wrap_command(self.command, self.working_dir, self.env_vars)

    def hostname_ref(self) -> str:
        """Get hostname reference for hetjob cross-component communication."""
        if self.het_group_index is None:
            return "127.0.0.1"  # Local fallback
        # For heterogeneous SLURM jobs, resolve nodelist to actual hostname
        return f"$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_{self.het_group_index} | head -n1)"

    def meta_ref(self, key: str) -> str:
        """Get metadata value (like port). Fails if key not found."""
        if key not in self.metadata:
            raise KeyError(
                f"Metadata key '{key}' not found in Command '{self.name}'. "
                f"Available keys: {list(self.metadata.keys())}"
            )
        return str(self.metadata[key])

    def prepare_for_execution(self, cluster_config: Dict) -> Tuple[str, Dict]:
        """Prepare command for execution.

        This method:
        1. Evaluates callables (resolves cross-component references)
        2. Wraps with installation_command if provided

        Returns:
            Tuple of (final_command, execution_config)
        """
        # 1. Evaluate if callable (for cross-component references like hostname_ref)
        if callable(self.command):
            result = self.command()

            if isinstance(result, tuple):
                final_command, runtime_metadata = result
                # Deep merge metadata, especially environment dict
                for key, value in runtime_metadata.items():
                    if key == "environment" and key in self.metadata:
                        # Merge environment dicts instead of replacing
                        self.metadata[key].update(value)
                    else:
                        self.metadata[key] = value
            else:
                final_command = result
        else:
            final_command = self.command

        # 2. Wrap with installation_command if provided
        if self.installation_command:
            final_command = install_packages_wrap(final_command, self.installation_command)

        # 3. Build execution config from metadata
        execution_config = {
            "num_tasks": self.metadata.get("num_tasks", 1),
            "num_gpus": self.metadata.get("gpus", self.gpus or 0),
            "num_nodes": self.metadata.get("nodes", self.nodes),
            "environment": self.metadata.get("environment", {}),
            "log_prefix": self.metadata.get("log_prefix", "main"),
            "mounts": self.metadata.get("mounts"),
            "container": self.metadata.get("container", self.container),  # Use container from metadata if available
        }

        return final_command, execution_config

    def get_name(self) -> str:
        return self.name


@dataclass
class HardwareConfig:
    """Hardware configuration for a group of tasks."""

    partition: Optional[str] = None
    qos: Optional[str] = None
    time_min: Optional[str] = None
    exclusive: bool = False
    num_gpus: Optional[int] = None
    num_nodes: Optional[int] = None


class CommandGroup:
    """Command group where commands run together with shared resource requirements."""

    def __init__(
        self,
        commands: List[Command],
        hardware: Optional[HardwareConfig] = None,
        name: Optional[str] = None,
        log_dir: Optional[str] = None,
    ):
        self.commands = commands
        self.hardware = hardware or HardwareConfig()
        self.name = name
        self.log_dir = log_dir


class Pipeline:
    """Top-level pipeline that composes command groups with dependency support.

    Jobs format: jobs=[{...}, {...}] - list of job dicts with dependencies and groups

    Dependency types:
    - Job dict objects: Internal dependencies on jobs in the same pipeline
    - Strings: External dependencies on other experiments
    """

    def __init__(
        self,
        name: str,
        cluster_config: Dict,
        jobs: List[Dict],
        reuse_code: bool = True,
        reuse_code_exp: Optional[str] = None,
        skip_hf_home_check: bool | None = None,
        with_ray: bool = False,
        run_after: Optional[Union[str, List[str]]] = None,  # Pipeline-level dependency on other experiments
    ):
        self.name = name
        self.cluster_config = cluster_config
        self.reuse_code = reuse_code
        self.reuse_code_exp = reuse_code_exp
        # If not explicitly set, resolve from cluster config (matching exp.py behavior)
        if skip_hf_home_check is None:
            skip_hf_home_check = cluster_config.get("skip_hf_home_check", False)
        self.skip_hf_home_check = skip_hf_home_check
        self.with_ray = with_ray
        self.run_after = run_after
        self.jobs = jobs

        # Validate configuration early
        self._validate()

        # Note: het_group_indices are assigned per-job in _plan_and_add_job, not globally

    def _validate(self):
        """Validate pipeline configuration early in __init__."""
        # Validate jobs
        if not self.jobs:
            raise ValueError("Pipeline requires at least one job")

        for idx, job_spec in enumerate(self.jobs):
            job_name = job_spec.get("name")
            if not job_name:
                raise ValueError(f"Job at index {idx} must have a 'name' field: {job_spec}")

        # Validate cluster_config has required fields
        if "executor" not in self.cluster_config:
            raise ValueError("cluster_config must have 'executor' field")
        if "containers" not in self.cluster_config:
            raise ValueError("cluster_config must have 'containers' field")

        # Validate HF_HOME if needed
        if self.cluster_config["executor"] != "none" and not self.skip_hf_home_check:
            env_vars = get_env_variables(self.cluster_config)
            if "HF_HOME" not in env_vars:
                raise RuntimeError(
                    "Invalid cluster_config: HF_HOME is missing from env_vars while skip_hf_home_check=False.\n"
                    f"Current env_vars: {self.cluster_config.get('env_vars', [])}\n"
                    "Please add a new variable: HF_HOME=/mounted/path/to/your/hf_home"
                )
            if not is_mounted_filepath(self.cluster_config, env_vars["HF_HOME"]):
                raise RuntimeError(f"Invalid cluster_config: HF_HOME={env_vars['HF_HOME']} is not a mounted path.")

    def run(self, dry_run: bool = False, log_dir: Optional[str] = None, _reuse_exp=None, sequential: bool = False):
        """Execute the pipeline by calling NeMo-Run directly.

        Args:
            dry_run: If True, validate without executing
            log_dir: Default log directory for groups that don't specify one (optional)
            _reuse_exp: Internal - reuse existing experiment object (for eval.py integration)
            sequential: If True, run tasks sequentially (only makes sense for local/none executors)
        """
        # Track job name -> task handle for dependency resolution
        job_name_to_handle = {}

        with get_exp(self.name, self.cluster_config, _reuse_exp) as exp:
            # Process each job in order
            for job_spec in self.jobs:
                job_name = job_spec["name"]  # Already validated in _validate()

                # Separate internal and external dependencies from the start
                # - Internal deps (task handles from current experiment) go to exp.add()
                # - External deps (SLURM job IDs from other experiments) go to executor
                internal_deps = []
                external_deps = []

                # Handle dependencies from job spec
                job_dependencies = job_spec.get("dependencies", [])
                # Handle explicit None (when dependencies key exists but value is None)
                if job_dependencies is None:
                    job_dependencies = []

                # If no job-level dependencies, apply pipeline-level run_after
                if not job_dependencies and self.run_after:
                    run_after_list = self.run_after if isinstance(self.run_after, list) else [self.run_after]
                    job_dependencies = run_after_list

                for dep in job_dependencies:
                    if isinstance(dep, str):
                        # String dependency = external experiment name
                        if self.cluster_config["executor"] == "slurm":
                            exp_handles = get_exp_handles(dep)
                            if len(exp_handles) == 0:
                                LOG.warning(
                                    f"No pending or running tasks found for experiment {dep}, cannot set dependencies."
                                )
                                # If no experiment found, treat as direct task handle (for _reuse_exp case)
                                if _reuse_exp:
                                    internal_deps.append(dep)
                                    LOG.info(
                                        f"Job '{job_name}' depends on task handle '{dep}' (from reused experiment)"
                                    )
                            else:
                                external_deps.extend(exp_handles)
                                LOG.info(
                                    f"Job '{job_name}' depends on external experiment '{dep}' ({len(exp_handles)} tasks)"
                                )
                        elif _reuse_exp:
                            # For non-SLURM executors with _reuse_exp, string deps are internal task handles
                            internal_deps.append(dep)
                            LOG.info(f"Job '{job_name}' depends on task handle '{dep}' (from reused experiment)")
                    elif isinstance(dep, dict):
                        # Dict dependency = internal job reference (by job spec object)
                        dep_name = dep.get("name")
                        if not dep_name:
                            raise ValueError(f"Job dependency must have a 'name' field: {dep}")
                        if dep_name in job_name_to_handle:
                            internal_deps.append(job_name_to_handle[dep_name])
                            LOG.info(
                                f"Job '{job_name}' depends on internal job '{dep_name}' (handle: {job_name_to_handle[dep_name]})"
                            )
                        else:
                            raise ValueError(
                                f"Job '{job_name}' depends on job '{dep_name}' which hasn't been processed yet. "
                                f"Make sure dependencies are listed before the jobs that depend on them in the jobs list."
                            )
                    else:
                        # Direct task handle object (not string or dict)
                        internal_deps.append(dep)
                        LOG.info(f"Job '{job_name}' depends on task handle (object)")

                # Convert empty lists to None for cleaner handling
                internal_deps = internal_deps if internal_deps else None
                external_deps = external_deps if external_deps else None

                # Check if this is a multi-group job or single group
                if "groups" in job_spec:
                    # If only one group in list, use single group job for efficiency
                    if len(job_spec["groups"]) == 1:
                        task_handle = self._add_single_group_job(
                            exp,
                            job_spec["groups"][0],
                            self.cluster_config,
                            default_log_dir=log_dir,
                            internal_deps=internal_deps,
                            external_deps=external_deps,
                        )
                    else:
                        # True multi-group: combine multiple groups into one heterogeneous SLURM job
                        task_handle = self._add_multi_group_job(
                            exp,
                            job_spec["groups"],
                            self.cluster_config,
                            default_log_dir=log_dir,
                            internal_deps=internal_deps,
                            external_deps=external_deps,
                        )
                elif "group" in job_spec:
                    # Single group job
                    task_handle = self._add_single_group_job(
                        exp,
                        job_spec["group"],
                        self.cluster_config,
                        default_log_dir=log_dir,
                        internal_deps=internal_deps,
                        external_deps=external_deps,
                    )
                else:
                    raise ValueError(f"Job spec must have either 'group' or 'groups': {job_spec}")

                # Track task handle for this job
                job_name_to_handle[job_name] = task_handle
                LOG.info(f"Added job '{job_name}' with task_handle={task_handle}")

            # Only run if not using existing experiment (matching generate_v0.py line 331)
            if not dry_run and not _reuse_exp:
                run_exp(exp, self.cluster_config, sequential=sequential)

                # Cache experiment for code reuse in future runs
                if self.cluster_config["executor"] != "none":
                    tunnel = get_tunnel(self.cluster_config)
                    cur_tunnel_hash = tunnel_hash(tunnel)
                    if cur_tunnel_hash not in REUSE_CODE_EXP:
                        REUSE_CODE_EXP[cur_tunnel_hash] = exp
                        LOG.info("Cached experiment for future code reuse")

            # When reusing experiment, return list of task handles (matching generate_v0.py line 335)
            if _reuse_exp:
                return list(job_name_to_handle.values())

            return exp

    def _prepare_command(self, command, cluster_config: Dict) -> Tuple[str, Dict]:
        """Prepare command and handle mpirun wrapping."""
        final_cmd, exec_config = command.prepare_for_execution(cluster_config)

        # Handle mpirun wrapping for non-SLURM executors
        num_tasks = exec_config["num_tasks"]
        if cluster_config["executor"] != "slurm" and num_tasks > 1:
            final_cmd = f"mpirun --allow-run-as-root -np {num_tasks} bash -c {shlex.quote(final_cmd)}"

        return final_cmd, exec_config

    def _resolve_container(self, exec_config: Dict, command, cluster_config: Dict) -> str:
        """Resolve container name to image path."""
        container_name = exec_config.get("container", command.container)
        if container_name in cluster_config.get("containers", {}):
            return cluster_config["containers"][container_name]
        return container_name

    def _create_executor(
        self,
        command,
        exec_config: Dict,
        container_image: str,
        cluster_config: Dict,
        log_dir: str,
        hardware: HardwareConfig,
        heterogeneous: bool,
        het_group: int,
        total_het_groups: int,
        overlap: bool,
        dependencies: Optional[List] = None,
    ):
        """Create executor with optional environment update."""
        env_context = (
            temporary_env_update(cluster_config, exec_config["environment"])
            if exec_config.get("environment")
            else nullcontext()
        )

        with env_context:
            return get_executor(
                cluster_config=cluster_config,
                container=container_image,
                num_nodes=exec_config["num_nodes"],
                tasks_per_node=exec_config["num_tasks"],
                gpus_per_node=exec_config["num_gpus"],
                job_name=command.name,
                log_dir=log_dir,
                log_prefix=exec_config["log_prefix"],
                partition=hardware.partition if hardware else None,
                qos=hardware.qos if hardware else None,
                time_min=hardware.time_min if hardware else None,
                heterogeneous=heterogeneous,
                het_group=het_group,
                total_het_groups=total_het_groups,
                overlap=overlap,
                mounts=exec_config.get("mounts"),
                with_ray=self.with_ray,
                slurm_kwargs={"exclusive": hardware.exclusive} if (hardware and hardware.exclusive) else None,
                dependencies=dependencies,
            )

    def _plan_and_add_job(
        self,
        exp,
        groups: List[CommandGroup],
        cluster_config: Dict,
        default_log_dir: Optional[str] = None,
        internal_deps: Optional[List] = None,
        external_deps: Optional[List] = None,
        heterogeneous: bool = False,
    ) -> str:
        """Plan commands/executors for one or more groups and add to experiment.

        This encapsulates shared logic between single-group and multi-group jobs. Behavior
        differences are controlled by the 'heterogeneous' flag and the provided 'groups'.

        Args:
            internal_deps: Task handles from same experiment (passed to exp.add())
            external_deps: SLURM job IDs from other experiments (passed to executor)
        """

        # Resolve log directory (use first group's log_dir if present)
        log_dir = groups[0].log_dir or default_log_dir
        if log_dir is None:
            raise ValueError(f"CommandGroup '{groups[0].name}' must have log_dir set, or provide it to pipeline.run()")

        commands: List[str] = []
        executors: List = []
        het_group_indices: List[int] = []

        # In heterogeneous jobs, collect environment from all commands for cross-component refs
        shared_env_vars: Dict[str, str] = {}
        if heterogeneous:
            for het_idx, group in enumerate(groups):
                for command in group.commands:
                    _, exec_config_probe = command.prepare_for_execution(cluster_config)
                    shared_env_vars.update(exec_config_probe.get("environment", {}))

        # Share packager across executors for efficiency (single-group only)
        shared_packager = None

        # Build commands and executors
        for het_idx, group in enumerate(groups):
            has_multiple_components = len(group.commands) > 1
            total_het_groups = (
                len(groups) if heterogeneous else (len(group.commands) if has_multiple_components else 1)
            )

            # For single-group jobs with multiple components, allow job-level GPU override for sbatch allocation
            job_level_gpus = (
                group.hardware.num_gpus if (not heterogeneous and has_multiple_components and group.hardware) else None
            )

            for comp_idx, command in enumerate(group.commands):
                # Assign het_group_index ONLY for heterogeneous jobs (per-job, not global)
                # Non-heterogeneous jobs use localhost, so het_group_index should remain None
                if heterogeneous:
                    command.het_group_index = het_idx
                else:
                    command.het_group_index = None

                final_cmd, exec_config = self._prepare_command(command, cluster_config)
                commands.append(final_cmd)

                # Adjust GPU allocation (first component gets job-level GPUs for sbatch) for single-group jobs
                exec_config["num_gpus"] = exec_config["num_gpus"] or 0
                if (not heterogeneous) and (comp_idx == 0) and (job_level_gpus is not None):
                    exec_config["num_gpus"] = job_level_gpus

                # Merge shared environment for heterogeneous jobs
                if heterogeneous and shared_env_vars:
                    exec_config["environment"].update(shared_env_vars)

                # Resolve container and create executor
                container_image = self._resolve_container(exec_config, command, cluster_config)
                # Pass external dependencies only to the first executor (SLURM doesn't support per-component dependencies in hetjobs)
                exec_dependencies = external_deps if (het_idx == 0 and comp_idx == 0) else None
                executor = self._create_executor(
                    command,
                    exec_config,
                    container_image,
                    cluster_config,
                    log_dir,
                    group.hardware,
                    heterogeneous,
                    het_idx if heterogeneous else comp_idx,
                    total_het_groups,
                    (len(group.commands) > 1),
                    dependencies=exec_dependencies,
                )

                # Share packager across executors for single-group jobs
                if not heterogeneous:
                    if comp_idx == 0 and het_idx == 0:
                        shared_packager = executor.packager
                    else:
                        executor.packager = shared_packager

                executors.append(executor)
                if heterogeneous:
                    het_group_indices.append(het_idx)

        # For heterogeneous jobs, set het_group_indices on the first executor
        if heterogeneous and executors:
            executors[0].het_group_indices = het_group_indices

        # Handle code reuse from previous experiments (single-group only)
        if (not heterogeneous) and cluster_config["executor"] != "none":
            tunnel = get_tunnel(cluster_config)
            if self.reuse_code:
                reuse_exp = self.reuse_code_exp or REUSE_CODE_EXP.get(tunnel_hash(tunnel))
                if reuse_exp is not None:
                    if isinstance(reuse_exp, str):
                        try:
                            reuse_exp = run.Experiment.from_id(reuse_exp)
                        except Exception:
                            try:
                                reuse_exp = run.Experiment.from_title(reuse_exp)
                            except Exception:
                                LOG.warning(f"Failed to load experiment {reuse_exp} for code reuse")
                                reuse_exp = None
                    if reuse_exp is not None:
                        LOG.info(f"Trying to reuse code from experiment {reuse_exp._title}")
                        reuse_key = get_packaging_job_key(reuse_exp._id, "nemo-run")
                        if reuse_key in reuse_exp.tunnels[tunnel.key].packaging_jobs:
                            reuse_dir = reuse_exp.tunnels[tunnel.key].packaging_jobs[reuse_key].dst_path
                            for executor in executors:
                                executor.packager.symlink_from_remote_dir = reuse_dir
                            LOG.info(f"Successfully reused code from {reuse_key}")
                        else:
                            LOG.warning(f"Relevant packaging job not found for experiment {reuse_exp._title}")
            else:
                # If reuse_code=False, clear cache
                REUSE_CODE_EXP.pop(tunnel_hash(tunnel), None)

        # Handle executor="none" path replacements (single-group only)
        if (not heterogeneous) and cluster_config["executor"] == "none":
            for idx in range(len(commands)):
                commands[idx] = commands[idx].replace(
                    "/nemo_run/code/nemo_skills", str(get_registered_external_repo("nemo_skills").path)
                )
                commands[idx] = commands[idx].replace("/nemo_run/code", "./")

        # Ray metadata handling
        if self.with_ray and cluster_config["executor"] == "slurm":
            metadata = {"use_with_ray_cluster": True}
        else:
            metadata = None

        # Add to experiment and return task ID
        # Note: Internal dependencies (task handles from same experiment) go to exp.add()
        #       External dependencies (SLURM job IDs from other experiments) go to executor
        if (not heterogeneous) and len(commands) == 1:
            task_id = exp.add(
                run.Script(inline=commands[0], metadata=metadata),
                executor=executors[0],
                name="nemo-run",
                dependencies=internal_deps,
            )
        else:
            task_id = exp.add(
                [
                    run.Script(inline=cmd, metadata=(metadata if idx == 0 else None))
                    for idx, cmd in enumerate(commands)
                ],
                executor=executors,
                name="nemo-run",
                dependencies=internal_deps,
            )

        return task_id

    def _add_single_group_job(
        self,
        exp,
        group: CommandGroup,
        cluster_config: Dict,
        default_log_dir: Optional[str] = None,
        internal_deps: Optional[List] = None,
        external_deps: Optional[List] = None,
    ) -> str:
        """Add a single CommandGroup as one job and return its task handle."""

        return self._plan_and_add_job(
            exp=exp,
            groups=[group],
            cluster_config=cluster_config,
            default_log_dir=default_log_dir,
            internal_deps=internal_deps,
            external_deps=external_deps,
            heterogeneous=False,
        )

    def _add_multi_group_job(
        self,
        exp,
        groups: List[CommandGroup],
        cluster_config: Dict,
        default_log_dir: Optional[str] = None,
        internal_deps: Optional[List] = None,
        external_deps: Optional[List] = None,
    ) -> str:
        """Add multiple CommandGroups as a single heterogeneous SLURM job and return task handle."""

        return self._plan_and_add_job(
            exp=exp,
            groups=groups,
            cluster_config=cluster_config,
            default_log_dir=default_log_dir,
            internal_deps=internal_deps,
            external_deps=external_deps,
            heterogeneous=True,
        )
