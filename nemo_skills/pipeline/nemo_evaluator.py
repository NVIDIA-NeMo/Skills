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
NeMo Evaluator (https://github.com/NVIDIA-NeMo/Evaluator) is a powerful
evaluation framework (external to NeMo-Skills) featuring numerous harnesses
with advanced configurability and uniform control (independent of harnesses via
request-response adapters). This module leverages both its advantages and the
capabilities of NeMo-Skills w.r.t. advanced orchestration, sandboxing etc.

### Architecture Overview:

NeMo-Skills                                                NeMo Evaluator
┌─────────────────────────────────┐                       ┌──────────────────────────────┐
│                                 │                       │                              │
│  Pipeline                       │                       │  NeMo Evaluator Launcher     │
│  ┌───────────────────────────┐  │                       │  ┌────────────────────────┐  │
│  │                           │  │                       │  │                        │  │
│  │  nemo_evaluator()         │◄─┼───────────────────────┼──────►                    │  │
│  │                           │  │                       │  │  - Configuration mgmt  │  │
│  │                           │  │ - Loads eval config   │  │  - Task definitions    │  │
│  │                           │  │ - Gets task metadata  │  │  - Container images    │  │
│  │                           │  │ - Constructs commands │  │  - Task mappings       │  │
│  │                           │  │ - Gets container IDs  │  │                        │  │
│  └───────────┬───────────────┘  │                       │  └────────────────────────┘  │
│              │                  │                       │                              │
│              ▼                  │                       │                              │
│  ┌───────────────────────────┐  │                       │                              │
│  │                           │  │                       │                              │
│  │  Command / CommandGroup   │  │                       │                              │
│  │                           │  │                       │                              │
│  │  - Main Server (vLLM)     │  │                       │                              │
│  │  - Judge Server (vLLM)    │  │                       │                              │
│  │  - Evaluator Client       │  │                       │                              │
│  │                           │  │                       │                              │
│  └───────────┬───────────────┘  │                       │                              │
│              │                  │                       │                              │
│              ▼                  │                       │  NeMo Evaluator Container    │
│  ┌───────────────────────────┐  │                       │  ┌────────────────────────┐  │
│  │                           │  │                       │  │                        │  │
│  │  NeMo-Run                 │  │                       │  │  - Runs evaluations    │  │
│  │                           │  │                       │  │  - Task-specific       │  │
│  │  - Submits jobs           ┼──┼───────────────────────┼──►  - Container depends   │  │
│  │  - Orchestrates execution │  │                       │  │    on task type        │  │
│  │                           │  │                       │  │                        │  │
│  └───────────────────────────┘  │                       │  └────────────────────────┘  │
│                                 │                       │                              │
│                                 │                       │                              │
│                                 │                       │                              │
│                                 │                       │                              │
└─────────────────────────────────┘                       └──────────────────────────────┘

 Flow:
   1. Pipeline loads evaluation config via NeMo Evaluator Launcher
   2. Gets task metadata, container images, and task mappings
   3. Constructs Command objects (servers + client) and groups them
   4. NeMo-Run executes commands, launching NeMo Evaluator Container
   5. Container runs task-specific evaluations

### Component Types:

- Evaluator Client:
   - Runs in NeMo Evaluator container using the command prepared by the Nemo Evaluator
   - Connects to main/judge servers via runtime URLs provided via
     declarative API.
   - Executes evaluation tasks available via NeMo Evaluator
- Main Server (optional): can be self-hosted or external
- Judge Server (optional): same

### Command Grouping Strategy:

- Both servers hosted: Separate groups (main+client, judge-only)

- Single/no servers: Single group with all components
  - co-host all containers on one node, shared resources

"""

import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import typer
from nemo_evaluator_launcher.api import RunConfig
from nemo_evaluator_launcher.common.helpers import get_eval_factory_command
from nemo_evaluator_launcher.common.mapping import get_task_from_mapping, load_tasks_mapping
from omegaconf import DictConfig, OmegaConf

import nemo_skills.pipeline.utils as pipeline_utils
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils.commands import vllm_server_command
from nemo_skills.pipeline.utils.declarative import Command, CommandGroup, HardwareConfig, Pipeline
from nemo_skills.utils import get_logger_name, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def nemo_evaluator(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help=(
            "One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
            "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument."
        ),
    ),
    output_dir: str = typer.Option(..., help="Where to put logs and .done tracking files for the run"),
    expname: str = typer.Option("nemo-evaluator", help="Nemo run experiment name"),
    # Orchestration knobs
    job_gpus: int = typer.Option(0, help="GPUs to allocate for the evaluator client when no servers are hosted"),
    job_nodes: int = typer.Option(1, help="Nodes to allocate for the evaluator job"),
    partition: str = typer.Option(None, help="Cluster partition to use"),
    qos: str = typer.Option(None, help="Slurm QoS"),
    mount_paths: str = typer.Option(None, help="Comma separated list of paths to mount on the remote machine"),
    log_dir: str = typer.Option(None, help="Custom location for logs"),
    exclusive: bool = typer.Option(False, help="If set will add exclusive flag to the slurm job."),
    with_sandbox: bool = typer.Option(False, help="If True, will start a sandbox container alongside this job"),
    keep_mounts_for_sandbox: bool = typer.Option(
        False,
        help=(
            "If True, will keep the mounts for the sandbox container. Risky; sandbox executes LLM commands and "
            "could potentially lead to data loss."
        ),
    ),
    # Optional self-hosted server (declarative) ? if server_type is set, a server will be co-scheduled
    server_type: Optional[str] = typer.Option(
        None,
        help=("If set, self-host a server and co-schedule with evaluator. Supported values: vllm (preferred)."),
    ),
    server_model: Optional[str] = typer.Option(
        None, help="Model path/name to serve when self-hosting (e.g., Qwen/Qwen3-4B-Thinking-2507)"
    ),
    server_gpus: int = typer.Option(0, help="GPUs to allocate for the self-hosted server (0 = no server)"),
    server_nodes: int = typer.Option(1, help="Nodes to allocate for the self-hosted server"),
    server_port: Optional[int] = typer.Option(None, help="Port for the server; if unset uses free/random"),
    server_args: Optional[str] = typer.Option(None, help="Extra args for server (passed through)"),
    server_entrypoint: Optional[str] = typer.Option(None, help="Custom entrypoint for server (advanced)"),
    server_container: Optional[str] = typer.Option(
        None, help="Container key/image for server (defaults to cluster 'nemo-skills' if None)"
    ),
    server_base_url: Optional[str] = typer.Option(
        None, help="Use an externally hosted server instead of self-hosting (e.g., http://host:port)"
    ),
    server_api_path: str = typer.Option("/v1/chat/completions", help="API path used for evaluator target url"),
    server_health_path: str = typer.Option("/health", help="Health path used to wait for server readiness"),
    # Optional judge self-hosted server (similar to main server)
    judge_server_type: Optional[str] = typer.Option(
        None,
        help=("If set, self-host a judge server and co-schedule with evaluator. Supported values: vllm (preferred)."),
    ),
    judge_server_model: Optional[str] = typer.Option(
        None, help="Model path/name to serve for judge (e.g., Qwen/Qwen3-32B-Instruct)"
    ),
    judge_server_gpus: int = typer.Option(0, help="GPUs to allocate for the judge server (0 = no judge server)"),
    judge_server_nodes: int = typer.Option(1, help="Nodes to allocate for the judge server"),
    judge_server_port: Optional[int] = typer.Option(None, help="Port for the judge server; if unset uses free/random"),
    judge_server_args: Optional[str] = typer.Option(None, help="Extra args for judge server (passed through)"),
    judge_server_entrypoint: Optional[str] = typer.Option(None, help="Custom entrypoint for judge server (advanced)"),
    judge_server_container: Optional[str] = typer.Option(
        None, help="Container key/image for judge server (defaults to cluster 'vllm' or 'nemo-skills')"
    ),
    judge_server_base_url: Optional[str] = typer.Option(
        None, help="Use an externally hosted judge server instead of self-hosting (e.g., http://host:port)"
    ),
    judge_server_api_path: str = typer.Option("/v1/chat/completions", help="API path used for judge target url"),
    judge_server_health_path: str = typer.Option(
        "/health", help="Health path used to wait for judge server readiness"
    ),
    # Experiment lifecycle and dependencies
    reuse_code: bool = typer.Option(True, help="If True, will reuse code from the provided experiment"),
    reuse_code_exp: str = typer.Option(None, help="If specified, reuse code from this experiment"),
    run_after: List[str] = typer.Option(None, help="List of expnames that must complete before this starts"),
    dependent_jobs: int = typer.Option(0, help="Launch this number of dependent jobs"),
    # Config discovery
    config_dir: str = typer.Option(None, help="Where to search for cluster configs"),
    dry_run: bool = typer.Option(False, help="If True, validate without submitting the job"),
    # Evaluator mapping/config knobs
    nemo_evaluator_config: str = typer.Option(
        help=(
            "Path to nemo-evaluator-launcher config YAML, see "
            "https://docs.nvidia.com/nemo/evaluator/latest/libraries/nemo-evaluator-launcher/configuration/index.html for documentation."
        ),
    ),
):
    """Run Nemo Evaluator tasks via nemo-skills orchestration.

    This function orchestrates NeMo Evaluator tasks by:
    1. Loading the evaluator configuration and task mappings
    2. For each task, determining server hosting strategy (self-hosted vs external)
    3. Building Command objects for servers (if needed) and evaluator client
    4. Grouping commands into CommandGroups based on hosting strategy
    5. Creating a Pipeline with all jobs and executing it

    The function supports four server hosting scenarios:
    - No servers: Client uses external URLs or config defaults
    - Main server only: Self-host main server, co-schedule with client
    - Judge server only: Self-host judge server, co-schedule with client
    - Both servers: Self-host both, create separate groups for resource allocation

    Returns:
        Pipeline execution result (experiment object or task handles)

    Example:
        Run evaluation with self-hosted main server:
        ```bash
        ns nemo-evaluator \\
            --nemo-evaluator-config configs/eval.yaml \\
            --output-dir /workspace/results \\
            --server-type vllm \\
            --server-model meta/llama-3.1-8b-instruct \\
            --server-gpus 8
        ```

        Use external servers:
        ```bash
        ns nemo-evaluator \\
            --nemo-evaluator-config configs/eval.yaml \\
            --output-dir /workspace/results \\
            --server-base-url http://main-server:8000 \\
            --judge-server-base-url http://judge-server:9000
        ```
    """
    setup_logging(disable_hydra_logs=False, use_rich=True)

    extra_overrides = f"{' '.join(ctx.args)}"
    LOG.info("Starting nemo_evaluator job")
    LOG.info("Extra overrides passed to evaluator: %s", extra_overrides)

    # Prepare cluster config and mount paths
    cluster_config = pipeline_utils.get_cluster_config(cluster, config_dir)
    cluster_config = pipeline_utils.resolve_mount_paths(cluster_config, mount_paths, create_remote_dir=False)

    if not log_dir:
        log_dir = f"{output_dir}/{expname}/nemo-evaluator-logs"

    # Validate mounts for output dir
    output_dir, log_dir = pipeline_utils.check_mounts(
        cluster_config,
        log_dir=log_dir,
        mount_map={output_dir: None},
    )

    # Create log directory for local executors (not created by check_mounts when check_mounted_paths=False)
    # Use create_remote_directory to create on host side (not inside container) to avoid permission issues
    if cluster_config.get("executor") in ["local", "none"]:
        pipeline_utils.create_remote_directory(log_dir, cluster_config)

    # Load evaluator configuration and task mappings
    launcher_run_cfg = RunConfig.from_hydra(
        config_dir=str(Path(nemo_evaluator_config).parent),
        config_name=str(Path(nemo_evaluator_config).stem),
        hydra_overrides=list(ctx.args),
    )

    # Build jobs for each task in the evaluator config
    tasks_mapping: dict[tuple[str, str], dict] = load_tasks_mapping()
    base_output_root = (output_dir or "").rstrip("/") if output_dir else None
    jobs = []
    for idx, task in enumerate(launcher_run_cfg.evaluation.tasks):
        task_definition = get_task_from_mapping(task.name, tasks_mapping)

        # Collect environment variables (global + task-specific)
        env_vars: Dict[str, str] = copy.deepcopy(dict(launcher_run_cfg.evaluation.get("env_vars", {})))
        env_vars.update(task.get("env_vars", {}))
        eval_image = task.get("container") or task_definition["container"]

        # Determine server hosting strategy
        hosting_server = bool(server_type) and (server_gpus or 0) > 0 and bool(server_model)
        with_external_server = (not hosting_server) and bool(server_base_url)
        hosting_judge = bool(judge_server_type) and (judge_server_gpus or 0) > 0 and bool(judge_server_model)
        with_external_judge = (not hosting_judge) and bool(judge_server_base_url)

        task_ctx = _TaskCreationContext(
            expname=expname,
            idx=idx,
            task_name=task.name,
            launcher_run_cfg=launcher_run_cfg,
            task_cfg=task,
            task_definition=task_definition,
            base_output_root=base_output_root,
            eval_image=eval_image,
            env_vars=env_vars,
            hosting_server=hosting_server,
            hosting_judge=hosting_judge,
            with_external_server=with_external_server,
            with_external_judge=with_external_judge,
            server_api_path=server_api_path,
            server_health_path=server_health_path,
            judge_server_api_path=judge_server_api_path,
            judge_server_health_path=judge_server_health_path,
            server_model=server_model,
            judge_server_model=judge_server_model,
            server_type=server_type,
            judge_server_type=judge_server_type,
            server_port=server_port,
            judge_server_port=judge_server_port,
            server_gpus=server_gpus,
            server_nodes=server_nodes,
            judge_server_gpus=judge_server_gpus,
            judge_server_nodes=judge_server_nodes,
            server_args=server_args,
            judge_server_args=judge_server_args,
            server_entrypoint=server_entrypoint,
            judge_server_entrypoint=judge_server_entrypoint,
            server_container=server_container,
            judge_server_container=judge_server_container,
            server_base_url=server_base_url,
            judge_server_base_url=judge_server_base_url,
            job_gpus=job_gpus,
            job_nodes=job_nodes,
            cluster_config=cluster_config,
            partition=partition,
            qos=qos,
            exclusive=exclusive,
        )

        # Build Command objects for each component
        main_server_cmd = _build_main_server_if_needed(task_ctx)
        judge_server_cmd = _build_judge_server_if_needed(task_ctx)
        client_cmd = _build_client_command(task_ctx, main_server_cmd, judge_server_cmd)

        # Group commands based on hosting strategy:
        # - Both servers: Separate groups (main+client, judge-only) for independent scaling
        # - Single/no servers: Single group with all components
        if main_server_cmd and judge_server_cmd:
            # Both servers: Create separate groups for independent resource allocation
            # Group 1: Main server + client (client references judge via cross-component refs)
            # Group 2: Judge server only (referenced by client in group 1)
            groups_for_job = [
                CommandGroup(
                    commands=[main_server_cmd, client_cmd],
                    hardware=_hardware_for_group(
                        task_ctx.partition,
                        task_ctx.server_gpus or None,
                        task_ctx.server_nodes or 1,
                        task_ctx.qos,
                        task_ctx.exclusive,
                    ),
                    name=f"{task_ctx.expname}-server-{task_ctx.idx}",
                    log_dir=log_dir,
                ),
                CommandGroup(
                    commands=[judge_server_cmd],
                    hardware=_hardware_for_group(
                        task_ctx.partition,
                        task_ctx.judge_server_gpus or None,
                        task_ctx.judge_server_nodes or 1,
                        task_ctx.qos,
                        task_ctx.exclusive,
                    ),
                    name=f"{task_ctx.expname}-judge-server-{task_ctx.idx}",
                    log_dir=log_dir,
                ),
            ]
        else:
            # Single or no servers: All components in one group
            sg_cmds: List[Command] = []
            if main_server_cmd:
                sg_cmds.append(main_server_cmd)
            if judge_server_cmd:
                sg_cmds.append(judge_server_cmd)
            sg_cmds.append(client_cmd)

            # Determine hardware allocation for the group
            if task_ctx.hosting_server or task_ctx.hosting_judge:
                # Use server GPUs if any servers are hosted
                total_server_gpus = (task_ctx.server_gpus or 0) + (task_ctx.judge_server_gpus or 0)
                group_num_gpus = total_server_gpus or None
                group_num_nodes = max(
                    task_ctx.server_nodes or 1, task_ctx.judge_server_nodes or 1, task_ctx.job_nodes or 1
                )
            else:
                # No servers: use job-level GPU allocation
                group_num_gpus = task_ctx.job_gpus or None
                group_num_nodes = task_ctx.job_nodes or 1

            groups_for_job = [
                CommandGroup(
                    commands=sg_cmds,
                    hardware=_hardware_for_group(
                        task_ctx.partition, group_num_gpus, group_num_nodes, task_ctx.qos, task_ctx.exclusive
                    ),
                    name=f"{task_ctx.expname}-{task_ctx.idx}",
                    log_dir=log_dir,
                )
            ]

        jobs.append({"name": f"{task_ctx.expname}-{task_ctx.idx}", "groups": groups_for_job})

    # Create and execute the pipeline
    pipeline = Pipeline(
        name=expname,
        cluster_config=cluster_config,
        jobs=jobs,
        reuse_code=reuse_code,
        reuse_code_exp=reuse_code_exp,
        skip_hf_home_check=True,  # avoid HF_HOME requirement for this orchestration path
        with_ray=False,
        run_after=run_after,
    )

    # Use sequential execution for local/none executors
    sequential = True if cluster_config.get("executor") in ["local", "none"] else False

    result = pipeline.run(dry_run=dry_run, sequential=sequential)
    return result


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    app()


def _create_serving_command_obj(
    *,
    cluster_config: dict,
    is_judge: bool,
    server_type: Optional[str],
    model: Optional[str],
    port: Optional[int],
    gpus: int,
    nodes: int,
    args: Optional[str],
    entrypoint: Optional[str],
    container: Optional[str],
    expname: str,
    idx: int,
    task_name: str,
) -> Command:
    """Create a Command object for a hosted serving component (main or judge server).

    This function wraps vllm_server_command and standardizes container selection,
    logging prefixes, and metadata for both main and judge servers.

    Args:
        cluster_config: Cluster configuration dictionary
        is_judge: True for judge server, False for main server
        server_type: Server type (currently only "vllm" supported)
        model: Model identifier to serve
        port: Server port (auto-assigned if None)
        gpus: Number of GPUs to allocate
        nodes: Number of nodes to allocate
        args: Extra arguments for server command
        entrypoint: Custom entrypoint override (advanced)
        container: Container image (defaults to cluster config)
        expname: Experiment name for naming
        idx: Task index for naming
        task_name: Task name for naming

    Returns:
        Command object configured for the serving component
    """
    stype = (server_type or "vllm").lower()
    sargs = args or ""
    if stype != "vllm":
        LOG.warning("Only vllm server_type is supported currently; got %s", stype)

    cmd_str, meta = vllm_server_command(
        cluster_config=cluster_config,
        model=model,  # type: ignore[arg-type]
        port=port,
        server_type=stype,
        gpus=gpus,
        nodes=nodes,
        args=sargs,
        entrypoint=entrypoint,
    )

    # Resolve container fallback when not explicitly provided
    if not container:
        container = cluster_config["containers"][stype]

    log_prefix = "judge-server" if is_judge else "server"
    name_role = "judge-server" if is_judge else "server"

    return Command(
        command=cmd_str,
        container=container,
        gpus=gpus,
        nodes=nodes or 1,
        name=f"{expname}-{name_role}-{idx}-{task_name}",
        metadata={
            **meta,
            "gpus": gpus,
            "log_prefix": log_prefix,
        },
    )


@dataclass
class _TaskCreationContext:
    """Local helper to pass around the information about the task and easier logic sharing."""

    expname: str
    idx: int
    task_name: str
    launcher_run_cfg: RunConfig
    task_cfg: DictConfig
    task_definition: dict
    base_output_root: Optional[str]
    eval_image: str
    env_vars: Dict[str, str]
    hosting_server: bool
    hosting_judge: bool
    with_external_server: bool
    with_external_judge: bool
    server_api_path: str
    server_health_path: str
    judge_server_api_path: str
    judge_server_health_path: str
    server_model: Optional[str]
    judge_server_model: Optional[str]
    server_type: Optional[str]
    judge_server_type: Optional[str]
    server_port: Optional[int]
    judge_server_port: Optional[int]
    server_gpus: int
    server_nodes: int
    judge_server_gpus: int
    judge_server_nodes: int
    server_args: Optional[str]
    judge_server_args: Optional[str]
    server_entrypoint: Optional[str]
    judge_server_entrypoint: Optional[str]
    server_container: Optional[str]
    judge_server_container: Optional[str]
    server_base_url: Optional[str]
    judge_server_base_url: Optional[str]
    job_gpus: int
    job_nodes: int
    cluster_config: Dict
    partition: Optional[str]
    qos: Optional[str]
    exclusive: bool


def _hardware_for_group(
    partition: Optional[str], num_gpus: Optional[int], num_nodes: int, qos: Optional[str], exclusive: bool
) -> HardwareConfig:
    """Create HardwareConfig for a CommandGroup.

    Args:
        partition: SLURM partition name
        num_gpus: Number of GPUs (None means no GPU allocation)
        num_nodes: Number of nodes
        qos: SLURM QoS setting
        exclusive: Whether to request exclusive node access (currently disabled due to SLURM issues)

    Returns:
        HardwareConfig instance with specified settings
    """
    return HardwareConfig(
        partition=partition,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        sbatch_kwargs={
            "qos": qos,
            # TODO(agronskiy): this results in the "invalid exclusive specification on squeue"
            # "exclusive": exclusive,
        },
    )


def _build_main_server_if_needed(ctx: _TaskCreationContext) -> Optional[Command]:
    """Build Command for main server if self-hosting is enabled.

    Returns:
        Command object for main server, or None if not hosting
    """
    if not ctx.hosting_server:
        return None
    return _create_serving_command_obj(
        cluster_config=ctx.cluster_config,
        is_judge=False,
        server_type=ctx.server_type,
        model=ctx.server_model,
        port=ctx.server_port,
        gpus=ctx.server_gpus,
        nodes=ctx.server_nodes,
        args=ctx.server_args,
        entrypoint=ctx.server_entrypoint,
        container=ctx.server_container,
        expname=ctx.expname,
        idx=ctx.idx,
        task_name=ctx.task_name,
    )


def _build_judge_server_if_needed(ctx: _TaskCreationContext) -> Optional[Command]:
    """Build Command for judge server if self-hosting is enabled.

    Returns:
        Command object for judge server, or None if not hosting
    """
    if not ctx.hosting_judge:
        return None
    return _create_serving_command_obj(
        cluster_config=ctx.cluster_config,
        is_judge=True,
        server_type=ctx.judge_server_type,
        model=ctx.judge_server_model,
        port=ctx.judge_server_port,
        gpus=ctx.judge_server_gpus,
        nodes=ctx.judge_server_nodes,
        args=ctx.judge_server_args,
        entrypoint=ctx.judge_server_entrypoint,
        container=ctx.judge_server_container,
        expname=ctx.expname,
        idx=ctx.idx,
        task_name=ctx.task_name,
    )


def _build_client_command(
    ctx: _TaskCreationContext, main_server_cmd: Optional[Command], judge_server_cmd: Optional[Command]
) -> Command:
    """Build Command for evaluator client.

    The client command behavior depends on server hosting:
    - If servers are co-hosted: Uses lambda factory to resolve runtime URLs via hostname_ref/meta_ref
    - If using external servers: Uses static URLs from server_base_url/judge_server_base_url
    - If no servers: Uses URLs from evaluator config or defaults

    Args:
        ctx: Task creation context with all configuration
        main_server_cmd: Main server Command if self-hosted, None otherwise
        judge_server_cmd: Judge server Command if self-hosted, None otherwise

    Returns:
        Command object for evaluator client
    """
    if ctx.hosting_server or ctx.hosting_judge:
        # Co-hosted servers: Use lambda factory to resolve runtime URLs
        # The lambda is evaluated at execution time when het_group_index is assigned
        def _client_cmd_factory():
            waits: List[str] = []
            target_url: Optional[str] = None
            judge_url: Optional[str] = None

            # Build main server URL from runtime references
            if ctx.hosting_server and main_server_cmd is not None:
                server_host = main_server_cmd.hostname_ref()
                server_port_val = main_server_cmd.meta_ref("port")
                base_url = f"http://{server_host}:{server_port_val}"
                waits.append(pipeline_utils.get_server_wait_cmd(f"{base_url}{ctx.server_health_path}"))
                target_url = f"{base_url}{ctx.server_api_path}"

            # Build judge server URL from runtime references
            if ctx.hosting_judge and judge_server_cmd is not None:
                jhost = judge_server_cmd.hostname_ref()
                jport = judge_server_cmd.meta_ref("port")
                jbase = f"http://{jhost}:{jport}"
                waits.append(pipeline_utils.get_server_wait_cmd(f"{jbase}{ctx.judge_server_health_path}"))
                judge_url = f"{jbase}{ctx.judge_server_api_path}"

            # Wait for servers to be ready, then run evaluator
            wait_cmd = " && ".join(waits) if waits else "true"
            cmd = _build_task_cmd(
                task_name=ctx.task_name,
                launcher_run_cfg=ctx.launcher_run_cfg,
                task_cfg=ctx.task_cfg,
                task_definition=ctx.task_definition,
                expname=ctx.expname,
                base_output_root=ctx.base_output_root,
                url_override=target_url,
                model_id=ctx.server_model,
                judge_url_override=judge_url,
                judge_model_id=ctx.judge_server_model,
            )
            return f"{wait_cmd} && {cmd}"

        return Command(
            command=_client_cmd_factory,
            container=ctx.eval_image,
            gpus=ctx.job_gpus or None,
            nodes=ctx.job_nodes or 1,
            name=f"{ctx.expname}-client-{ctx.idx}-{ctx.task_name}",
            metadata={
                "log_prefix": "main",
                "environment": ctx.env_vars,
                "gpus": ctx.job_gpus or None,
            },
        )

    # No hosted servers: Use external URLs or config defaults
    server_url = None
    if ctx.with_external_server and ctx.server_base_url:
        server_url = ctx.server_base_url.rstrip("/") + ctx.server_api_path
    judge_url = None
    if ctx.with_external_judge and ctx.judge_server_base_url:
        judge_url = ctx.judge_server_base_url.rstrip("/") + ctx.judge_server_api_path

    eval_cmd = _build_task_cmd(
        task_name=ctx.task_name,
        launcher_run_cfg=ctx.launcher_run_cfg,
        task_cfg=ctx.task_cfg,
        task_definition=ctx.task_definition,
        expname=ctx.expname,
        base_output_root=ctx.base_output_root,
        url_override=server_url,
        model_id=ctx.server_model,
        judge_url_override=judge_url,
        judge_model_id=ctx.judge_server_model,
    )

    return Command(
        command=eval_cmd,
        container=ctx.eval_image,
        gpus=None,
        nodes=ctx.job_nodes or 1,
        name=f"{ctx.expname}-{ctx.idx}-{ctx.task_name}",
        metadata={
            "log_prefix": "main",
            "environment": ctx.env_vars,
            "gpus": ctx.job_gpus or None,
        },
    )


def _build_task_cmd(
    task_name: str,
    launcher_run_cfg: DictConfig,
    task_cfg: DictConfig,
    task_definition: dict,
    expname: str,
    base_output_root: Optional[str],
    url_override: Optional[str] = None,
    model_id: Optional[str] = None,
    judge_url_override: Optional[str] = None,
    judge_model_id: Optional[str] = None,
) -> str:
    """Construct the evaluator command string with runtime URL overrides.

    This function builds the nemo-evaluator-launcher command for a specific task,
    injecting runtime URLs for main and judge servers via Hydra overrides.

    Args:
        task_name: Task identifier (e.g., "ifeval", "gpqa_diamond")
        launcher_run_cfg: Global evaluator configuration from RunConfig
        task_cfg: Task-specific configuration (may include task-level overrides)
        task_definition: Task definition from mapping (container, harness info)
        expname: Experiment name for output directory structure
        base_output_root: Base directory for task outputs
        url_override: Main server URL to inject (for co-hosted or external servers)
        model_id: Main model ID to inject
        judge_url_override: Judge server URL to inject (for co-hosted or external judge)
        judge_model_id: Judge model ID to inject

    Returns:
        Complete shell command string ready to execute

    Note:
        URL overrides are injected via Hydra's override mechanism:
        - Main: target.api_endpoint.url
        - Judge: config.params.extra.judge.url
        Output directory is set to: {base_output_root}/{expname}/nemo-evaluator-results/{task_name}
    """
    task_cfg_copy = copy.deepcopy(task_cfg)
    if url_override:
        OmegaConf.update(task_cfg_copy, "overrides", {"target.api_endpoint.url": url_override}, force_add=True)

    if model_id:
        OmegaConf.update(
            task_cfg_copy,
            "overrides",
            {"target.api_endpoint.model_id": model_id},
            force_add=True,
        )

    if judge_url_override or judge_model_id:
        if judge_url_override:
            OmegaConf.update(
                task_cfg_copy,
                "overrides",
                {"config.params.extra.judge.url": judge_url_override},
                force_add=True,
            )
        if judge_model_id:
            OmegaConf.update(
                task_cfg_copy,
                "overrides",
                {"config.params.extra.judge.model_id": judge_model_id},
                force_add=True,
            )

    if base_output_root:
        task_out = f"{base_output_root}/{expname}/nemo-evaluator-results/{task_name}"
        OmegaConf.update(task_cfg_copy, "overrides", {"config.output_dir": task_out}, force_add=True)

    cmd_struct = get_eval_factory_command(launcher_run_cfg, task_cfg_copy, task_definition)

    return cmd_struct.cmd
