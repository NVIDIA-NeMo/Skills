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

from __future__ import annotations

import logging
import os
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

if TYPE_CHECKING:
    from nemo_skills.pipeline.backends.base import JobSpec

import nemo_run as run

from nemo_skills.pipeline.utils import (
    get_env_variables,
    get_executor,
    get_exp,
    get_exp_handles,
    get_registered_external_repo,
    get_tunnel,
    run_exp,
    temporary_env_update,
)
from nemo_skills.pipeline.utils.exp import (
    REUSE_CODE_EXP,
    get_packaging_job_key,
    tunnel_hash,
)
from nemo_skills.pipeline.utils.mounts import is_mounted_filepath
from nemo_skills.pipeline.utils.server import wrap_python_path
from nemo_skills.utils import get_logger_name

# Import backend types for Kubernetes support (lazy import in method to avoid circular deps)
# from nemo_skills.pipeline.backends import get_backend, JobSpec, ContainerSpec, ResourceSpec, JobStatus

"""
Simplified declarative pipeline system using Command with run.Script objects.

Basic Example (Single job with multiple commands):
    from nemo_skills.pipeline.utils.scripts import ServerScript, SandboxScript, GenerationClientScript
    from nemo_skills.pipeline.utils.declarative import Command, CommandGroup, HardwareConfig, Pipeline

    # Create Script objects for server and sandbox
    # Scripts handle port allocation, cross-component references, and command building
    server_script = ServerScript(
        server_type="vllm",
        model_path="Qwen/Qwen2.5-Math-7B-Instruct",
        server_args="--tensor-parallel-size 1"
    )
    sandbox_script = SandboxScript()

    # Create generation client that references server and sandbox
    # Cross-component references (hostname_ref, port) are resolved at runtime
    client_script = GenerationClientScript(
        output_dir="/results/inference",
        extra_arguments="++prompt_config=math ++split=test",
        servers=[server_script],  # References server for hostname/port
        model_names=["Qwen/Qwen2.5-Math-7B-Instruct"],
        server_types=["vllm"],
        sandbox=sandbox_script,  # References sandbox for port
        with_sandbox=True,
    )

    # Wrap Scripts in Commands with container and resource info
    server = Command(script=server_script, container="vllm", name="server")
    sandbox = Command(script=sandbox_script, container="nemo-skills", name="sandbox")
    client = Command(script=client_script, container="nemo-skills", name="client")

    # Group them together (they run in one SLURM job)
    inference_group = CommandGroup(
        commands=[server, sandbox, client],
        hardware=HardwareConfig(partition="batch", num_gpus=1),
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
    from nemo_skills.pipeline.utils.scripts import ServerScript, SandboxScript, GenerationClientScript
    from nemo_run import Script

    log_dir = "/experiments/full_pipeline/logs"

    # Job 1: Preprocessing with custom Script
    @dataclass(kw_only=True)
    class PreprocessScript(Script):
        input_file: str
        output_file: str

        def __post_init__(self):
            cmd = f"python preprocess.py --input {self.input_file} --output {self.output_file}"
            self.inline = cmd
            object.__setattr__(self, 'entrypoint', 'bash')

    preprocess_script = PreprocessScript(
        input_file="data.jsonl",
        output_file="processed.jsonl"
    )
    preprocess = Command(script=preprocess_script, name="preprocess")
    prep_group = CommandGroup(
        commands=[preprocess],
        hardware=HardwareConfig(partition="cpu"),
        name="prep",
        log_dir=log_dir
    )
    prep_job = {"name": "prep", "group": prep_group}

    # Job 2: Two different model servers (HETEROGENEOUS SLURM job with 2 het groups)
    # 8B model group
    server_8b = ServerScript(
        server_type="vllm",
        model_path="Qwen/Qwen2.5-Math-7B-Instruct",
        server_args="--tensor-parallel-size 1"
    )
    sandbox_8b = SandboxScript()
    client_8b = GenerationClientScript(
        output_dir="/results/eval_8b",
        extra_arguments="++prompt_config=math",
        servers=[server_8b],
        model_names=["Qwen/Qwen2.5-Math-7B-Instruct"],
        server_types=["vllm"],
        sandbox=sandbox_8b,
        with_sandbox=True,
    )

    group_8b = CommandGroup(
        commands=[
            Command(script=server_8b, container="vllm", name="server_8b"),
            Command(script=sandbox_8b, container="nemo-skills", name="sandbox_8b"),
            Command(script=client_8b, container="nemo-skills", name="eval_8b"),
        ],
        hardware=HardwareConfig(partition="batch", num_gpus=1),
        name="eval_8b",
        log_dir=log_dir
    )

    # 32B model group
    server_32b = ServerScript(
        server_type="vllm",
        model_path="Qwen/Qwen2.5-Math-32B-Instruct",
        server_args="--tensor-parallel-size 4"
    )
    sandbox_32b = SandboxScript()
    client_32b = GenerationClientScript(
        output_dir="/results/eval_32b",
        extra_arguments="++prompt_config=math",
        servers=[server_32b],
        model_names=["Qwen/Qwen2.5-Math-32B-Instruct"],
        server_types=["vllm"],
        sandbox=sandbox_32b,
        with_sandbox=True,
    )

    group_32b = CommandGroup(
        commands=[
            Command(script=server_32b, container="vllm", name="server_32b"),
            Command(script=sandbox_32b, container="nemo-skills", name="sandbox_32b"),
            Command(script=client_32b, container="nemo-skills", name="eval_32b"),
        ],
        hardware=HardwareConfig(partition="batch", num_gpus=4),
        name="eval_32b",
        log_dir=log_dir
    )

    evals_job = {"name": "evals", "groups": [group_8b, group_32b], "dependencies": [prep_job]}

    # Job 3: Report generation (depends on both evaluations)
    @dataclass(kw_only=True)
    class ReportScript(Script):
        output_file: str

        def __post_init__(self):
            self.inline = f"python generate_report.py --output {self.output_file}"
            object.__setattr__(self, 'entrypoint', 'bash')

    report_script = ReportScript(output_file="report.txt")
    report = Command(script=report_script, name="report")
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

LOG = logging.getLogger(get_logger_name(__file__))


def _sanitize_k8s_name(name: str, max_length: int = 63) -> tuple[str, bool]:
    """Sanitize a name to be Kubernetes-compliant.

    Kubernetes naming requirements:
    - Must be lowercase
    - Only alphanumeric characters and hyphens
    - Must start and end with alphanumeric character
    - Maximum 63 characters

    Note: This only handles format conversion. Uniqueness (to prevent job name
    collisions) is handled by the K8s backend, similar to how local backend
    adds UUID suffixes.

    Args:
        name: The original name to sanitize.
        max_length: Maximum length (default 63 for K8s).

    Returns:
        Tuple of (sanitized_name, was_modified) where was_modified is True
        if the name was changed during sanitization.
    """
    original = name

    # Convert to lowercase
    name = name.lower()

    # Replace invalid characters with hyphens (underscores, slashes, dots, etc.)
    name = re.sub(r"[^a-z0-9-]", "-", name)

    # Collapse multiple consecutive hyphens into one
    name = re.sub(r"-+", "-", name)

    # Strip leading/trailing hyphens
    name = name.strip("-")

    # Truncate to max length, but don't end with a hyphen
    if len(name) > max_length:
        name = name[:max_length].rstrip("-")

    # Handle empty result
    if not name:
        name = "job"

    was_modified = name != original
    return name, was_modified


@dataclass
class Command:
    """Declarative command for running tasks in containers using run.Script objects.

    Example:
        server = ServerScript(server_type="vllm", model_path="/models/llama", ...)
        Command(script=server, container="vllm", name="my_server")
    """

    script: run.Script
    container: str = "nemo-skills"
    name: str = "command"

    def prepare_for_execution(self, cluster_config: Dict) -> Tuple[run.Script, Dict]:
        """Prepare script for execution.

        This method:
        1. Evaluates lazy commands (if script.inline is callable)
        2. Builds execution config from Script fields

        Returns:
            Tuple of (Script_object, execution_config)
        """
        runtime_metadata = {}

        # If script.inline is callable (lazy command building), evaluate it now
        if callable(self.script.inline):
            result = self.script.inline()

            if isinstance(result, tuple):
                evaluated_command, runtime_metadata = result
            else:
                evaluated_command = result

            # Update script.inline with evaluated command
            self.script.set_inline(evaluated_command)

        # Build execution config from Script fields
        execution_config = {
            "log_prefix": getattr(self.script, "log_prefix", "main"),
            "environment": runtime_metadata.get("environment", {}),
            "mounts": None,  # Mounts not currently exposed by Scripts
            "container": self.container,
        }

        # Return the Script object itself
        return self.script, execution_config

    def get_name(self) -> str:
        return self.name


@dataclass
class HardwareConfig:
    """Hardware configuration for a group of tasks.

    Attributes:
        partition: Slurm partition or K8s resource pool name.
        num_gpus: Number of GPUs per node.
        num_nodes: Number of nodes for distributed jobs.
        num_tasks: Number of tasks (processes) per node.
        cpus: CPU cores per node/request for Kubernetes (defaults to heuristic from num_tasks).
        memory_request_gb: Memory request in GB for K8s scheduling. None = auto-calculate.
        memory_limit_gb: Memory limit in GB. None = no limit (pod can burst).
        sbatch_kwargs: Additional Slurm sbatch arguments.

    Memory Behavior (Kubernetes):
        - memory_request_gb: Reserved for scheduling (default: 16GB + 32GB per GPU)
        - memory_limit_gb: Max usage cap (default: None = no limit, can use available)

        This ensures proper scheduling while allowing GPU workloads to burst.
    """

    partition: Optional[str] = None
    num_gpus: Optional[int] = None
    num_nodes: Optional[int] = None
    num_tasks: Optional[int] = 1
    cpus: Optional[int] = None
    memory_request_gb: Optional[float] = None  # None = auto-calculate based on GPUs
    memory_limit_gb: Optional[float] = None  # None = no limit
    sbatch_kwargs: Optional[dict] = None


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

    def _make_unique_job_name(self, base_name: str, max_length: int = 63) -> str:
        """Generate a unique job name by adding a timestamp+pid suffix.

        This ensures job names are unique across re-runs, which is required for
        Kubernetes (job names must be unique in namespace) and useful for Slurm
        (makes it easier to distinguish job runs in logs/monitoring).

        The suffix format is -XXXXXYYYY (10 chars total), where:
        - XXXXX: timestamp modulo 100000
        - YYYY: process ID modulo 10000

        Args:
            base_name: The original job name (should already be sanitized for K8s).
            max_length: Maximum total length (default 63 for K8s compatibility).

        Returns:
            Unique job name with timestamp+pid suffix, truncated if necessary.
        """
        suffix = f"-{int(time.time()) % 100000:05d}{os.getpid() % 10000:04d}"
        suffix_len = len(suffix)  # 10 characters

        # Truncate base name to leave room for suffix
        max_base_len = max_length - suffix_len
        if len(base_name) > max_base_len:
            base_name = base_name[:max_base_len].rstrip("-")

        return f"{base_name}{suffix}"

    def run(self, dry_run: bool = False, log_dir: Optional[str] = None, _reuse_exp=None, sequential: bool = False):
        """Execute the pipeline on the appropriate backend.

        Routes to Kubernetes backend for executor='kubernetes', otherwise uses
        the existing NeMo-Run based execution for Slurm/local backends.

        Args:
            dry_run: If True, validate without executing
            log_dir: Default log directory for groups that don't specify one (optional)
            _reuse_exp: Internal - reuse existing experiment object (for eval.py integration)
            sequential: If True, run tasks sequentially (only makes sense for local/none executors)
        """
        executor = self.cluster_config["executor"]

        # Route to Kubernetes backend for kubernetes executor
        if executor == "kubernetes":
            return self._run_kubernetes(dry_run=dry_run, log_dir=log_dir, sequential=sequential)

        # Use existing NeMo-Run based execution for Slurm/local
        return self._run_nemo_run(dry_run=dry_run, log_dir=log_dir, _reuse_exp=_reuse_exp, sequential=sequential)

    def _run_nemo_run(
        self, dry_run: bool = False, log_dir: Optional[str] = None, _reuse_exp=None, sequential: bool = False
    ):
        """Execute the pipeline using NeMo-Run (for Slurm/local backends).

        This is the existing implementation, extracted to a separate method.
        """
        # Track job name -> task handle for dependency resolution
        job_name_to_handle = {}

        with get_exp(self.name, self.cluster_config, _reuse_exp) as exp:
            # Process each job in order
            for job_spec in self.jobs:
                original_job_name = job_spec["name"]  # Already validated in _validate()
                # Generate unique name for submission (consistent with K8s pathway)
                job_name = self._make_unique_job_name(original_job_name)
                LOG.info(f"Job '{original_job_name}' will be submitted as '{job_name}'")

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
                        try:
                            dep_name = dep["name"]
                        except KeyError as exc:
                            raise ValueError(f"Job dependency must have a 'name' field: {dep}") from exc
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

                # Track task handle by ORIGINAL name for dependency lookups
                job_name_to_handle[original_job_name] = task_handle
                LOG.info(f"Added job '{original_job_name}' (as '{job_name}') with task_handle={task_handle}")

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

    def _run_kubernetes(self, dry_run: bool = False, log_dir: Optional[str] = None, sequential: bool = False):
        """Execute the pipeline using Kubernetes backend.

        This method converts Pipeline jobs to JobSpecs and submits them
        to the Kubernetes backend.

        Args:
            dry_run: If True, validate and show what would run without executing
            log_dir: Default log directory for groups that don't specify one
            sequential: If True, wait for each job to complete before starting the next
        """
        # Lazy import to avoid circular dependencies
        from nemo_skills.pipeline.backends import (
            JobStatus,
            get_backend,
        )

        backend = get_backend(self.cluster_config)
        LOG.info(f"Using Kubernetes backend (namespace: {self.cluster_config.get('namespace', 'default')})")

        # Check if any jobs have dependencies - if so, auto-enable sequential mode
        # K8s Jobs don't have native dependency support like Slurm's afterok
        has_dependencies = any(job_spec.get("dependencies") for job_spec in self.jobs)
        if has_dependencies and not sequential:
            LOG.warning(
                "Pipeline has job dependencies but sequential=False. "
                "Kubernetes does not support native job dependencies (like Slurm's afterok). "
                "Auto-enabling sequential mode to ensure correct execution order."
            )
            sequential = True

        # Track job name -> JobHandle for dependency resolution (keyed by ORIGINAL name)
        job_name_to_handle = {}

        # Collect all job specs for dry run display
        job_specs = []

        for job_spec in self.jobs:
            original_job_name = job_spec["name"]  # Already validated in _validate()

            # Sanitize job name for Kubernetes compliance
            sanitized_name, was_modified = _sanitize_k8s_name(original_job_name)
            if was_modified:
                LOG.warning(
                    f"Job name '{original_job_name}' is not Kubernetes-compliant. "
                    f"Sanitized to '{sanitized_name}'. "
                    f"(K8s requires: lowercase, alphanumeric and hyphens only, max 63 chars)"
                )

            # Add unique suffix (consistent with NeMo-Run pathway)
            # _make_unique_job_name handles truncation to keep total <= 63 chars
            job_name = self._make_unique_job_name(sanitized_name)
            LOG.info(f"Job '{original_job_name}' will be submitted as '{job_name}'")

            # Get the groups for this job (same structure as _run_nemo_run)
            if "groups" in job_spec:
                groups = job_spec["groups"]
            elif "group" in job_spec:
                groups = [job_spec["group"]]
            else:
                raise ValueError(f"Job spec must have either 'group' or 'groups': {job_spec}")

            # Handle dependencies (mirroring _run_nemo_run structure)
            dependency_handles = []
            job_dependencies = job_spec.get("dependencies", [])
            if job_dependencies is None:
                job_dependencies = []

            # If no job-level dependencies, apply pipeline-level run_after (same as _run_nemo_run)
            if not job_dependencies and self.run_after:
                run_after_list = self.run_after if isinstance(self.run_after, list) else [self.run_after]
                job_dependencies = run_after_list

            for dep in job_dependencies:
                if isinstance(dep, str):
                    # String dependency = external experiment name - not supported on K8s
                    raise ValueError(
                        f"External string dependency '{dep}' not supported on Kubernetes. "
                        "Use internal dict dependencies (job names) instead."
                    )
                elif isinstance(dep, dict):
                    # Dict dependency = internal job reference (same as _run_nemo_run)
                    try:
                        dep_name = dep["name"]
                    except KeyError as exc:
                        raise ValueError(f"Job dependency must have a 'name' field: {dep}") from exc
                    if dep_name in job_name_to_handle:
                        dependency_handles.append(job_name_to_handle[dep_name])
                        LOG.info(f"Job '{original_job_name}' depends on internal job '{dep_name}'")
                    else:
                        raise ValueError(
                            f"Job '{original_job_name}' depends on job '{dep_name}' which hasn't been processed yet. "
                            f"Make sure dependencies are listed before the jobs that depend on them in the jobs list."
                        )
                else:
                    # Direct handle object (same as _run_nemo_run)
                    dependency_handles.append(dep)
                    LOG.info(f"Job '{original_job_name}' depends on handle (object)")

            # Convert handles to job IDs for K8s (handles may be None in dry-run)
            dependency_job_ids = (
                [h.job_id for h in dependency_handles if h is not None] if dependency_handles else None
            )
            if dependency_job_ids is not None and len(dependency_job_ids) == 0:
                dependency_job_ids = None

            # Convert CommandGroups to JobSpec
            k8s_job_spec = self._convert_groups_to_job_spec(
                job_name=job_name,
                groups=groups,
                log_dir=log_dir,
                dependencies=dependency_job_ids,
            )
            job_specs.append((job_name, k8s_job_spec))

            if dry_run:
                self._print_dry_run_job(job_name, k8s_job_spec)
                # Track job name so dependency resolution works in dry-run mode
                job_name_to_handle[original_job_name] = None
                continue

            # Submit the job
            handle = backend.submit_job(k8s_job_spec)
            # Track by ORIGINAL name for dependency lookups
            job_name_to_handle[original_job_name] = handle
            LOG.info(f"Submitted job '{original_job_name}' as '{job_name}' (job_id: {handle.job_id})")

            # If sequential, wait for this job to complete before continuing
            if sequential:
                LOG.info(f"Waiting for job '{job_name}' to complete (sequential mode)...")
                status = backend.wait_for_completion(handle)
                LOG.info(f"Job '{job_name}' completed with status: {status.value}")
                if status != JobStatus.SUCCEEDED:
                    raise RuntimeError(f"Job '{job_name}' did not succeed (status={status.value}), aborting pipeline")

        if dry_run:
            LOG.info("Dry run complete. No jobs were submitted.")
            return None

        # Return handles for monitoring
        return job_name_to_handle

    def _convert_groups_to_job_spec(
        self,
        job_name: str,
        groups: List[CommandGroup],
        log_dir: Optional[str] = None,
        dependencies: Optional[List[str]] = None,
    ) -> JobSpec:
        """Convert CommandGroups to a Kubernetes JobSpec.

        All commands from all groups are converted to containers in a single
        multi-container Pod (Kubernetes equivalent of Slurm heterogeneous jobs).

        For multi-node jobs (num_nodes > 1 in HardwareConfig), the JobSpec will
        have num_nodes > 1 which tells the KubernetesBackend to create an
        Indexed Job with a Headless Service for distributed training.

        Args:
            job_name: Name for the job
            groups: List of CommandGroups to convert
            log_dir: Log directory (for environment variables)
            dependencies: List of job IDs this job depends on

        Returns:
            JobSpec ready for submission to KubernetesBackend
        """
        from nemo_skills.pipeline.backends import ContainerSpec, JobSpec, ResourceSpec

        containers = []
        node_selector = None
        num_nodes = 1

        # Set backend on all scripts before processing
        for group in groups:
            for command in group.commands:
                command.script.backend = "kubernetes"
                # het_group_index not needed for K8s (all containers share localhost)
                command.script.het_group_index = None

        for group in groups:
            # Get node selector from resource pool if specified
            if group.hardware and group.hardware.partition:
                resource_pools = self.cluster_config.get("resource_pools", {})
                if group.hardware.partition in resource_pools:
                    pool_config = resource_pools[group.hardware.partition]
                    node_selector = pool_config.get("node_selector")

            # Track the maximum num_nodes across groups
            if group.hardware and group.hardware.num_nodes and group.hardware.num_nodes > num_nodes:
                num_nodes = group.hardware.num_nodes

            for command in group.commands:
                # Prepare the command (evaluates lazy commands)
                script, exec_config = self._prepare_command(command, self.cluster_config)

                # _prepare_command() resolves lazy callables; inline is expected to be a string now.
                cmd_str = script.inline
                if not isinstance(cmd_str, str):
                    raise TypeError(
                        f"Command '{command.name}' must resolve to a string inline command, got {type(cmd_str).__name__}"
                    )

                # Resolve container image
                container_image = self._resolve_container(exec_config, command, self.cluster_config)

                # Build resource spec
                # Memory request: auto-calculated if not specified (for K8s scheduling)
                # Memory limit: None by default (pods can use available memory)
                hardware = group.hardware or HardwareConfig()
                cpu_request = hardware.cpus
                if cpu_request is None:
                    cpu_floor = 4 if (hardware.num_gpus or 0) > 0 else 1
                    cpu_request = max(hardware.num_tasks or 0, cpu_floor)
                resources = ResourceSpec(
                    gpus=hardware.num_gpus or 0,
                    cpus=cpu_request,
                    memory_request_gb=hardware.memory_request_gb,  # None = auto-calculate
                    memory_limit_gb=hardware.memory_limit_gb,  # None = no limit
                )

                # Build environment variables
                # Note: Cluster-level env_vars are added by KubernetesBackend._build_container()
                # Here we only add command-specific env vars and log directory
                env_vars = {}
                env_vars.update(exec_config.get("environment", {}))
                if log_dir or group.log_dir:
                    env_vars["NEMO_LOG_DIR"] = log_dir or group.log_dir

                # Get ports from script if available
                ports = []
                if hasattr(script, "port"):
                    script_port = script.port
                    if isinstance(script_port, int) and 1 <= script_port <= 65535:
                        ports = [script_port]
                    elif script_port is not None:
                        LOG.warning(
                            "Ignoring invalid port value %r on command '%s'; expected int in [1, 65535]",
                            script_port,
                            command.name,
                        )

                # Create container spec
                container = ContainerSpec(
                    name=command.name,
                    image=container_image,
                    command=["bash", "-c", cmd_str],
                    env_vars=env_vars,
                    resources=resources,
                    ports=ports,
                )
                containers.append(container)

        # Parse timeout from cluster config
        timeout_str = self.cluster_config.get("default_timeout", "6h")
        timeout_seconds = self._parse_timeout(timeout_str)

        # Build labels
        labels = {
            "app": "nemo-skills",
            "pipeline": self.name,
        }

        return JobSpec(
            name=job_name,
            containers=containers,
            num_nodes=num_nodes,
            timeout_seconds=timeout_seconds,
            dependencies=dependencies,
            labels=labels,
            node_selector=node_selector,
        )

    def _parse_timeout(self, timeout_str: str) -> int:
        """Parse timeout string to seconds.

        Supports formats: '6h', '30m', '3600', '06:00:00'
        """
        timeout_str = timeout_str.strip().lower()

        if not timeout_str:
            return 6 * 3600  # Default 6 hours

        try:
            # Already seconds
            if timeout_str.isdigit():
                return int(timeout_str)

            # Hours format (e.g., '6h')
            if timeout_str.endswith("h"):
                return int(timeout_str[:-1]) * 3600

            # Minutes format (e.g., '30m')
            if timeout_str.endswith("m"):
                return int(timeout_str[:-1]) * 60

            # HH:MM:SS format
            if ":" in timeout_str:
                parts = timeout_str.split(":")
                if len(parts) == 3:
                    hours, minutes, seconds = map(int, parts)
                    return hours * 3600 + minutes * 60 + seconds
                if len(parts) == 2:
                    minutes, seconds = map(int, parts)
                    return minutes * 60 + seconds
        except ValueError as e:
            raise ValueError(f"Invalid timeout format: '{timeout_str}'") from e

        raise ValueError(
            f"Unrecognized timeout format: '{timeout_str}'. Supported formats: "
            "'3600' (seconds), '6h' (hours), '30m' (minutes), '06:00:00' (HH:MM:SS)"
        )

    def _print_dry_run_job(self, job_name: str, spec: JobSpec):
        """Print job details for dry run."""
        LOG.info(f"\n{'=' * 60}")
        LOG.info(f"Job: {job_name}")
        LOG.info(f"{'=' * 60}")
        LOG.info(f"Containers: {len(spec.containers)}")
        for container in spec.containers:
            LOG.info(f"  - {container.name}")
            LOG.info(f"    Image: {container.image}")
            LOG.info(f"    GPUs: {container.resources.gpus}")
            command_text = " ".join(container.command)
            max_chars = 200
            if len(command_text) > max_chars:
                command_text = f"{command_text[:max_chars]}..."
            LOG.info(f"    Command: {command_text}")
        if spec.dependencies:
            LOG.info(f"Dependencies: {spec.dependencies}")
        LOG.info(f"Timeout: {spec.timeout_seconds}s")

    def _prepare_command(self, command, cluster_config: Dict) -> Tuple[run.Script, Dict]:
        """Prepare command for execution.

        Returns:
            Tuple of (Script_object, exec_config)
        """
        script, exec_config = command.prepare_for_execution(cluster_config)
        # Only rewrite paths for "none" executor (native execution without containers)
        # For "local" executor (Docker), paths should stay as /nemo_run/code/... since
        # that's where the code is mounted inside the container
        if cluster_config.get("executor") == "none":
            script = self._rewrite_local_paths(script)
        # Note: mpirun wrapping for multi-task scripts is handled by the executor
        return script, exec_config

    def _rewrite_local_paths(self, script: run.Script) -> run.Script:
        """For executor='none', replace /nemo_run/code paths with local repo paths."""
        nemo_repo = get_registered_external_repo("nemo_skills")
        if nemo_repo is None:
            return script

        pkg_path = str(nemo_repo.path)
        repo_root = str(nemo_repo.path.parent)

        def _replace(cmd: str) -> str:
            return cmd.replace("/nemo_run/code/nemo_skills", pkg_path).replace("/nemo_run/code", repo_root)

        inline_cmd = script.inline
        if isinstance(inline_cmd, str):
            script.set_inline(_replace(inline_cmd))
        elif callable(inline_cmd):
            original_inline = inline_cmd

            def wrapped_inline():
                result = original_inline()
                if isinstance(result, tuple):
                    cmd, metadata = result
                    return _replace(cmd), metadata
                return _replace(result)

            script.set_inline(wrapped_inline)

        return script

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
        job_name_override: Optional[str] = None,
    ):
        """Create executor with optional environment update."""
        env_context = (
            temporary_env_update(cluster_config, exec_config["environment"])
            if exec_config.get("environment")
            else nullcontext()
        )

        # Check if the script should span all nodes from the group's HardwareConfig.
        # Scripts with span_group_nodes=True (e.g., ServerScript) use the group's num_nodes.
        # Scripts with span_group_nodes=False (default) run on 1 node - important for multi-node
        # setups with --overlap where client/sandbox should only run on the master node.
        span_group_nodes = getattr(command.script, "span_group_nodes", False)
        num_nodes = 1
        if span_group_nodes and hardware and hardware.num_nodes is not None:
            num_nodes = hardware.num_nodes

        with env_context:
            return get_executor(
                cluster_config=cluster_config,
                container=container_image,
                num_nodes=num_nodes,
                tasks_per_node=hardware.num_tasks if hardware and hardware.num_tasks is not None else 1,
                gpus_per_node=hardware.num_gpus if hardware and hardware.num_gpus is not None else 0,
                job_name=job_name_override if job_name_override else command.name,
                log_dir=log_dir,
                log_prefix=exec_config["log_prefix"],
                partition=hardware.partition if hardware else None,
                heterogeneous=heterogeneous,
                het_group=het_group,
                total_het_groups=total_het_groups,
                overlap=overlap,
                mounts=exec_config.get("mounts"),
                with_ray=self.with_ray,
                sbatch_kwargs=hardware.sbatch_kwargs,
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

        scripts: List[run.Script] = []
        executors: List = []
        het_group_indices: List[int] = []

        # Assign het_group_index and backend values before evaluating any commands so
        # cross-references (e.g., hostname_ref) see the correct values regardless of
        # processing order.
        backend = cluster_config["executor"]
        for het_idx, group in enumerate(groups):
            for command in group.commands:
                command.script.het_group_index = het_idx if heterogeneous else None
                command.script.backend = backend

        # Prepare commands once and collect runtime data for a second pass where we
        # construct executors. This ensures all scripts have resolved cross-references.
        prepared_commands: List[Dict] = []
        shared_env_vars: Dict[str, str] = {}

        for het_idx, group in enumerate(groups):
            has_multiple_components = len(group.commands) > 1
            total_het_groups = (
                len(groups) if heterogeneous else (len(group.commands) if has_multiple_components else 1)
            )

            for comp_idx, command in enumerate(group.commands):
                script, exec_config = self._prepare_command(command, cluster_config)

                if isinstance(script.inline, str):
                    if cluster_config.get("executor") not in ("none", "local"):
                        script.set_inline(wrap_python_path(script.inline))

                prepared_commands.append(
                    {
                        "het_idx": het_idx,
                        "comp_idx": comp_idx,
                        "group": group,
                        "command": command,
                        "script": script,
                        "exec_config": exec_config,
                        "total_het_groups": total_het_groups,
                        "overlap": len(group.commands) > 1,
                    }
                )

                if heterogeneous:
                    shared_env_vars.update(exec_config.get("environment", {}))

        # Share packager across executors for efficiency (single-group only)
        shared_packager = None

        # Build commands and executors using prepared data
        for entry in prepared_commands:
            het_idx = entry["het_idx"]
            comp_idx = entry["comp_idx"]
            group = entry["group"]
            command = entry["command"]
            script = entry["script"]
            exec_config = entry["exec_config"]
            total_het_groups = entry["total_het_groups"]
            overlap = entry["overlap"]

            scripts.append(script)

            # Merge shared environment for heterogeneous jobs
            if heterogeneous and shared_env_vars:
                exec_config["environment"].update(shared_env_vars)

            # Resolve container and create executor
            container_image = self._resolve_container(exec_config, command, cluster_config)
            # Pass external dependencies only to the first executor (SLURM doesn't support per-component dependencies in hetjobs)
            exec_dependencies = external_deps if (het_idx == 0 and comp_idx == 0) else None

            # Always use group.name for SLURM job name (consistent across all components)
            # The group name is set to task_name in generate.py, without component suffixes
            # Component names (like {task_name}_server, {task_name}_sandbox) are only used for log_prefix
            job_name_for_slurm = group.name

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
                overlap,
                dependencies=exec_dependencies,
                job_name_override=job_name_for_slurm,
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

        # Note: Path replacements for executor="none" are no longer needed with Script interface

        # Ray metadata handling
        if self.with_ray and cluster_config["executor"] == "slurm":
            metadata = {"use_with_ray_cluster": True}
        else:
            metadata = None

        # Add to experiment and return task ID
        # Note: Internal dependencies (task handles from same experiment) go to exp.add()
        #       External dependencies (SLURM job IDs from other experiments) go to executor
        if (not heterogeneous) and len(scripts) == 1:
            # Single script - pass directly to exp.add()
            if metadata:
                scripts[0].metadata = metadata
            task_id = exp.add(
                scripts[0],
                executor=executors[0],
                name="nemo-run",
                dependencies=internal_deps,
            )
        else:
            # Multiple scripts or heterogeneous job
            # Apply metadata to first script only
            if metadata:
                scripts[0].metadata = metadata

            task_id = exp.add(
                scripts,
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
