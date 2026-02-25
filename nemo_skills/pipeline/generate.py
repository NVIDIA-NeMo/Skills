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
import importlib
import logging
import os
from typing import Dict, List, Optional

import typer

import nemo_skills.pipeline.utils as pipeline_utils
from nemo_skills.dataset.utils import import_from_path
from nemo_skills.inference import GENERATION_MODULE_MAP, GenerationType
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils.cluster import parse_kwargs
from nemo_skills.pipeline.utils.declarative import (
    Command,
    CommandGroup,
    HardwareConfig,
    Pipeline,
)
from nemo_skills.pipeline.utils.scripts import (
    GenerationClientScript,
    SandboxScript,
    ServerScript,
)
from nemo_skills.utils import (
    compute_chunk_ids,
    get_logger_name,
    setup_logging,
    str_ids_to_list,
    validate_wandb_project_name,
)

LOG = logging.getLogger(get_logger_name(__file__))

# TODO: add num_jobs here for consistency with eval?


def _create_job_unified(
    models: List[str],
    server_configs: List[Optional[Dict]],
    generation_params: Dict,
    cluster_config: Dict,
    installation_command: Optional[str],
    with_sandbox: bool,
    partition: Optional[str],
    keep_mounts_for_sandbox: bool,
    task_name: str,
    log_dir: str,
    sbatch_kwargs: Optional[Dict] = None,
    sandbox_env_overrides: Optional[List[str]] = None,
) -> List[CommandGroup]:
    """
    Create CommandGroups for n models (unified for n=1 and n>1).

    Structure:
    - Group 0: Model 0 server + client + (optional sandbox)
    - Group 1: Model 1 server (if n>1)
    - Group N: Model N server (if n>1)

    For n=1, returns a single-element list. The Pipeline automatically
    optimizes single-group lists to efficient single-group jobs.

    Args:
        models: List of model paths
        server_configs: List of server configurations (one per model, None if not hosting)
        generation_params: Dict of parameters for generation (output_dir, etc.)
        cluster_config: Cluster configuration
        installation_command: Installation command to run before client
        with_sandbox: Whether to include sandbox
        partition: Slurm partition
        keep_mounts_for_sandbox: Whether to keep mounts for sandbox
        task_name: Name for the task
        log_dir: Directory for logs
        sbatch_kwargs: Additional sbatch kwargs

    Returns:
        List of CommandGroup objects (one per het group)
    """
    num_models = len(models)
    groups = []
    server_scripts = []  # Track server Script objects for cross-component references

    for model_idx, (model_path, server_config) in enumerate(zip(models, server_configs)):
        components = []
        server_script = None

        # Track GPU/node requirements for this group (from server config)
        group_gpus = 0
        group_nodes = 1

        # 1. Add server if needed
        if server_config is not None and int(server_config.get("num_gpus", 0)) > 0:
            server_type = server_config["server_type"]
            server_container = server_config.get("container") or cluster_config["containers"][server_type]

            # Create ServerScript
            server_script = ServerScript(
                server_type=server_type,
                model_path=server_config["model_path"],
                cluster_config=cluster_config,
                num_gpus=server_config["num_gpus"],
                num_nodes=server_config["num_nodes"],
                server_args=server_config.get("server_args", ""),
                server_entrypoint=server_config.get("server_entrypoint"),
                port=server_config.get("server_port"),
                allocate_port=(server_config.get("server_port") is None),
            )

            # Set group GPU/node requirements from server config
            group_gpus = server_config["num_gpus"]
            group_nodes = server_config["num_nodes"]

            server_cmd = Command(
                script=server_script,
                container=server_container,
                name=f"{task_name}_model_{model_idx}_server" if num_models > 1 else f"{task_name}_server",
            )
            components.append(server_cmd)
            server_scripts.append(server_script)
        else:
            # No server for this model (pre-hosted)
            server_scripts.append(None)

        # 2. Group 0 gets the client and sandbox
        if model_idx == 0:
            # Create sandbox script (if with_sandbox)
            sandbox_script = None
            if with_sandbox:
                sandbox_script = SandboxScript(
                    cluster_config=cluster_config,
                    keep_mounts=keep_mounts_for_sandbox,
                    allocate_port=True,  # Always allocate port for sandbox
                    env_overrides=sandbox_env_overrides,
                )

                sandbox_cmd = Command(
                    script=sandbox_script,
                    container=cluster_config["containers"]["sandbox"],
                    name=f"{task_name}_sandbox",
                )
                components.append(sandbox_cmd)

            # Create client script with cross-component references to all servers
            client_script = GenerationClientScript(
                output_dir=generation_params["output_dir"],
                input_file=generation_params.get("input_file"),
                input_dir=generation_params.get("input_dir"),
                extra_arguments=generation_params.get("extra_arguments", ""),
                random_seed=generation_params.get("random_seed"),
                chunk_id=generation_params.get("chunk_id"),
                num_chunks=generation_params.get("num_chunks"),
                preprocess_cmd=generation_params.get("preprocess_cmd"),
                postprocess_cmd=generation_params.get("postprocess_cmd"),
                wandb_parameters=generation_params.get("wandb_parameters"),
                with_sandbox=with_sandbox,
                script=generation_params.get("script", "nemo_skills.inference.generate"),
                requirements=generation_params.get("requirements"),
                # Multi-server support (works for single and multi-model)
                servers=server_scripts if server_scripts else None,
                server_addresses_prehosted=generation_params.get("server_addresses_prehosted"),
                model_names=generation_params.get("model_names"),
                server_types=generation_params.get("server_types"),
                sandbox=sandbox_script,
                installation_command=installation_command,
            )

            client_cmd = Command(
                script=client_script,
                container=cluster_config["containers"]["nemo-skills"],
                name=f"{task_name}",
            )
            components.append(client_cmd)

        # Only create group if it has components (skip empty groups for pre-hosted models)
        if components:
            group_tasks = server_script.num_tasks if (server_config and server_script) else 1

            group = CommandGroup(
                commands=components,
                hardware=HardwareConfig(
                    partition=partition,
                    num_gpus=group_gpus,
                    num_nodes=group_nodes,
                    num_tasks=group_tasks,
                    sbatch_kwargs=sbatch_kwargs,
                ),
                name=f"{task_name}_model_{model_idx}_group" if num_models > 1 else task_name,
                log_dir=log_dir,
            )
            groups.append(group)

    return groups


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def generate(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    input_file: str = typer.Option(
        None, help="Path to the input data file. Can either specify input_file or input_dir, but not both. "
    ),
    input_dir: str = typer.Option(
        None,
        help="Path to the input data directory. Can either specify input_file or input_dir, but not both. "
        "If input_file is not provided, will use output-rs{{seed}}.jsonl inside input_dir as input_files. "
        "In this case, the random seed parameter is used both for input and for output files, which "
        "means it's a 1-1 mapping (not 1-num_random_seeds as in the case of input_file).",
    ),
    output_dir: str = typer.Option(..., help="Where to put results"),
    expname: str = typer.Option("generate", help="Nemo run experiment name"),
    generation_type: GenerationType | None = typer.Option(None, help="Type of generation to perform"),
    generation_module: str = typer.Option(
        None,
        help="Path to the generation module to use. "
        "If not specified, will use the registered generation module for the "
        "generation type (which is required in this case).",
    ),
    model: List[str] = typer.Option(
        None,
        help="Path to the model(s). CLI: space-separated. Python API: string or list. "
        "Single value broadcasts to all models for multi-model generation.",
    ),
    server_address: List[str] = typer.Option(
        None,
        help="Server address(es). CLI: space-separated. Python API: string or list. "
        "Single value broadcasts to all models.",
    ),
    server_type: List[pipeline_utils.SupportedServers] = typer.Option(
        ...,
        help="Server type(s). CLI: space-separated. Python API: string or list. "
        "Single value broadcasts to all models.",
    ),
    server_gpus: List[int] = typer.Option(
        None,
        help="Number of GPUs per model. CLI: space-separated ints. Python API: int or list. "
        "Single value broadcasts to all models.",
    ),
    server_nodes: List[int] = typer.Option(
        [1],
        help="Number of nodes per model. CLI: space-separated ints. Python API: int or list. "
        "Single value broadcasts to all models.",
    ),
    server_args: List[str] = typer.Option(
        [""],
        help="Server arguments per model. CLI: space-separated. Python API: string or list. "
        "Single value broadcasts to all models.",
    ),
    server_entrypoint: List[str] = typer.Option(
        None,
        help="Server entrypoint(s). CLI: space-separated. Python API: string or list. "
        "Single value broadcasts to all models.",
    ),
    server_container: List[str] = typer.Option(
        None,
        help="Container image(s). CLI: space-separated. Python API: string or list. "
        "Single value broadcasts to all models.",
    ),
    dependent_jobs: int = typer.Option(0, help="Specify this to launch that number of dependent jobs"),
    mount_paths: str = typer.Option(None, help="Comma separated list of paths to mount on the remote machine"),
    num_random_seeds: int = typer.Option(
        None, help="Specify if want to run many generations with high temperature for the same input"
    ),
    random_seeds: str = typer.Option(
        None,
        help="List of random seeds to use for generation. Separate with , or .. to specify range. "
        "Can provide a list directly when using through Python",
    ),
    starting_seed: int = typer.Option(0, help="Starting seed for random sampling"),
    num_chunks: int = typer.Option(
        None,
        help="Number of chunks to split the dataset into. If None, will not chunk the dataset.",
    ),
    chunk_ids: str = typer.Option(
        None,
        help="List of explicit chunk ids to run. Separate with , or .. to specify range. "
        "Can provide a list directly when using through Python",
    ),
    preprocess_cmd: str = typer.Option(None, help="Command to run before generation"),
    postprocess_cmd: str = typer.Option(None, help="Command to run after generation"),
    partition: str = typer.Option(
        None, help="Can specify if need interactive jobs or a specific non-default partition"
    ),
    qos: str = typer.Option(None, help="Specify Slurm QoS, e.g. to request interactive nodes"),
    time_min: str = typer.Option(None, help="If specified, will use as a time-min slurm parameter"),
    run_after: List[str] = typer.Option(
        None, help="Can specify a list of expnames that need to be completed before this one starts"
    ),
    reuse_code: bool = typer.Option(
        True,
        help="If True, will reuse the code from the provided experiment. "
        "If you use it from Python, by default the code will be re-used from "
        "the last submitted experiment in the current Python session, so set to False to disable "
        "(or provide reuse_code_exp to override).",
    ),
    reuse_code_exp: str = typer.Option(
        None,
        help="If specified, will reuse the code from this experiment. "
        "Can provide an experiment name or an experiment object if running from code.",
    ),
    config_dir: str = typer.Option(None, help="Can customize where we search for cluster configs"),
    log_dir: str = typer.Option(None, help="Can specify a custom location for slurm logs."),
    exclusive: bool | None = typer.Option(None, help="If set will add exclusive flag to the slurm job."),
    rerun_done: bool = typer.Option(
        False, help="If True, will re-run jobs even if a corresponding '.done' file already exists"
    ),
    with_sandbox: bool = typer.Option(False, help="If True, will start a sandbox container alongside this job"),
    sandbox_env_overrides: List[str] = typer.Option(
        None,
        help="Extra environment variables for the sandbox container in KEY=VALUE format. "
        "E.g., --sandbox-env-overrides NEMO_SKILLS_SANDBOX_BLOCK_NETWORK=1 to enable network blocking.",
    ),
    keep_mounts_for_sandbox: bool = typer.Option(
        False,
        help="If True, will keep the mounts for the sandbox container. Note that, it is risky given that sandbox executes LLM commands and could potentially lead to data loss. So, we advise not to use this unless absolutely necessary.",
    ),
    check_mounted_paths: bool = typer.Option(False, help="Check if mounted paths are available on the remote machine"),
    log_samples: bool = typer.Option(
        False,
        help="If True, will log random samples from the output files to wandb. "
        "Requires WANDB_API_KEY to be set in the environment. "
        "Use wandb_name/wandb_group/wandb_project to specify where to log.",
    ),
    wandb_name: str = typer.Option(
        None,
        help="Name of the wandb group to sync samples to. If not specified, but log_samples=True, will use expname.",
    ),
    wandb_group: str = typer.Option(None, help="Name of the wandb group to sync samples to."),
    wandb_project: str = typer.Option(
        "nemo-skills",
        help="Name of the wandb project to sync samples to.",
    ),
    installation_command: str | None = typer.Option(
        None,
        help="An installation command to run before main job. Only affects main task (not server or sandbox). "
        "You can use an arbitrary command here and we will run it on a single rank for each node. "
        "E.g. 'pip install my_package'",
    ),
    skip_hf_home_check: bool | None = typer.Option(
        None,
        help="If True, skip checking that HF_HOME env var is defined in the cluster config.",
    ),
    dry_run: bool = typer.Option(False, help="If True, will not run the job, but will validate all arguments."),
    sbatch_kwargs: str = typer.Option(
        "",
        help="Additional sbatch kwargs to pass to the job scheduler. Values should be provided as a JSON string or as a `dict` if invoking from code.",
    ),
    _reuse_exp: str = typer.Option(None, help="Internal option to reuse an experiment object.", hidden=True),
    _task_dependencies: List[str] = typer.Option(
        None, help="Internal option to specify task dependencies.", hidden=True
    ),
):
    """Generate LLM completions for single or multiple models.

    Supports both single-model and multi-model generation through a unified interface.

    Parameter Types:
        Multi-model parameters (model, server_*, etc.) use List[T] type hints for Typer CLI
        compatibility, but accept both scalars and lists when called from Python:
        - CLI: --model m1 m2 (space-separated) → Typer converts to ["m1", "m2"]
        - Python API: model="m1" or model=["m1", "m2"] → Both work (normalized internally)
        - Single values broadcast to all models: server_gpus=8 → [8, 8, 8] for 3 models

    Multi-model usage requires either --generation-type or --generation-module.

    Run `python -m nemo_skills.inference.generate --help` for other supported arguments
    (need to be prefixed with ++, since we use Hydra for that script).
    """
    setup_logging(disable_hydra_logs=False, use_rich=True)
    extra_arguments = f"{' '.join(ctx.args)}"
    LOG.info("Starting generation job")
    LOG.info("Extra arguments that will be passed to the underlying script: %s", extra_arguments)

    # Normalize model configuration to list
    models_list = pipeline_utils.normalize_models_config(model)
    num_models = len(models_list)

    LOG.info(f"Number of models: {num_models}")
    for model_idx, model_name in enumerate(models_list):
        LOG.info(f"  Model {model_idx}: {model_name}")

    # Convert server_type enum values to strings
    def convert_server_type_to_string(server_type):
        return server_type.value if hasattr(server_type, "value") else server_type

    if isinstance(server_type, list):
        server_type = [convert_server_type_to_string(st) for st in server_type]
    else:
        server_type = convert_server_type_to_string(server_type)

    # Normalize all server parameters to per-model lists
    server_types_list = pipeline_utils.normalize_parameter(server_type, num_models, "server_type")
    server_gpus_list = pipeline_utils.normalize_parameter(server_gpus, num_models, "server_gpus")
    server_nodes_list = pipeline_utils.normalize_parameter(server_nodes, num_models, "server_nodes")
    server_args_list = pipeline_utils.normalize_parameter(server_args, num_models, "server_args")
    server_entrypoints_list = pipeline_utils.normalize_parameter(server_entrypoint, num_models, "server_entrypoint")
    server_containers_list = pipeline_utils.normalize_parameter(server_container, num_models, "server_container")

    if server_address is not None:
        server_addresses_list = pipeline_utils.normalize_parameter(server_address, num_models, "server_address")
    else:
        server_addresses_list = [None] * num_models

    # Validate multi-model requirements
    if num_models > 1:
        if generation_type is None and generation_module is None:
            raise ValueError(
                "Multi-model generation requires either --generation-type or --generation-module to be specified"
            )

    if log_samples:
        wandb_parameters = {
            "name": wandb_name or expname,
            "project": wandb_project,
            "group": wandb_group,
        }
        validate_wandb_project_name(
            wandb_project=wandb_project,
            wandb_name=wandb_name or expname,
            wandb_group=wandb_group,
        )
    else:
        wandb_parameters = None

    if random_seeds and num_random_seeds:
        raise ValueError("Cannot specify both random_seeds and num_random_seeds")
    if num_random_seeds:
        random_seeds = list(range(starting_seed, starting_seed + num_random_seeds))
    if isinstance(random_seeds, str):
        random_seeds = str_ids_to_list(random_seeds)

    if num_chunks:
        chunk_ids = compute_chunk_ids(chunk_ids, num_chunks)
    if chunk_ids is None:
        chunk_ids = [None]

    # Prepare cluster config and mount paths
    cluster_config = pipeline_utils.get_cluster_config(cluster, config_dir)
    cluster_config = pipeline_utils.resolve_mount_paths(
        cluster_config, mount_paths, create_remote_dir=check_mounted_paths
    )

    if not log_dir:
        log_dir = f"{output_dir}/generation-logs"

    output_dir, log_dir = pipeline_utils.check_mounts(
        cluster_config,
        log_dir=log_dir,
        mount_map={output_dir: None},
        check_mounted_paths=check_mounted_paths,
    )

    if generation_module is not None and generation_type is not None:
        raise ValueError("Cannot specify both generation_module and generation_type. ")
    if generation_module is None:
        generation_module = GENERATION_MODULE_MAP[generation_type or GenerationType.generate]

    if generation_module.endswith(".py") or os.sep in generation_module:
        path_suffix = ".py" if not generation_module.endswith(".py") else ""
        generation_task = import_from_path(generation_module + path_suffix)
    else:
        generation_task = importlib.import_module(generation_module)
    if not hasattr(generation_task, "GENERATION_TASK_CLASS"):
        raise ValueError(
            f"Module {generation_module} does not have a GENERATION_TASK_CLASS attribute. "
            "Please provide a valid generation module."
        )
    generation_task = generation_task.GENERATION_TASK_CLASS
    extra_arguments = f"{generation_task.get_generation_default_args()} {extra_arguments}"
    generation_requirements = generation_task.get_generation_requirements()
    extra_arguments_original = extra_arguments

    # Treat no random seeds as a single None seed to unify the code paths
    if not random_seeds:
        random_seeds = [None]

    remaining_jobs = pipeline_utils.get_remaining_jobs(
        cluster_config=cluster_config,
        output_dir=output_dir,
        random_seeds=random_seeds,
        chunk_ids=chunk_ids,
        rerun_done=rerun_done,
    )

    if _task_dependencies is None:
        _task_dependencies = []

    # Parse sbatch kwargs
    sbatch_kwargs = parse_kwargs(sbatch_kwargs, exclusive=exclusive, qos=qos, time_min=time_min)

    # Build jobs list using declarative interface
    jobs = []
    all_job_names = []

    for seed_idx, (seed, chunk_ids) in enumerate(remaining_jobs.items()):
        if wandb_parameters:
            # no need for chunks as it will run after merging
            wandb_parameters["samples_file"] = pipeline_utils.get_chunked_rs_filename(
                output_dir,
                random_seed=seed,
                chunk_id=None,
            )
        for chunk_id in chunk_ids:
            # Configure clients for each model
            server_configs = []
            server_addresses_resolved = []
            # For single model: configure_client returns extra_args with server config appended
            # For multi-model: use original extra_args (server config added as lists in get_generation_cmd)
            extra_arguments = extra_arguments_original

            for model_idx in range(num_models):
                get_random_port_for_server = pipeline_utils.should_get_random_port(
                    server_gpus_list[model_idx], exclusive
                )

                srv_config, srv_address, srv_extra_args = pipeline_utils.configure_client(
                    model=models_list[model_idx],
                    server_type=server_types_list[model_idx],
                    server_address=server_addresses_list[model_idx],
                    server_gpus=server_gpus_list[model_idx],
                    server_nodes=server_nodes_list[model_idx],
                    server_args=server_args_list[model_idx],
                    server_entrypoint=server_entrypoints_list[model_idx],
                    server_container=server_containers_list[model_idx],
                    extra_arguments=extra_arguments_original if model_idx == 0 else "",
                    get_random_port=get_random_port_for_server,
                )
                server_configs.append(srv_config)
                server_addresses_resolved.append(srv_address)

                # For single model, capture the extra_args with server config from configure_client
                if model_idx == 0 and num_models == 1:
                    extra_arguments = srv_extra_args

            # Base task name (shared across all dependent jobs in the chain)
            task_name = f"{expname}-rs{seed}" if seed is not None else expname
            if chunk_id is not None:
                task_name += f"-chunk{chunk_id}"

            # Handle dependent_jobs chain
            dependencies = _task_dependencies.copy() if _task_dependencies else []
            prev_job = None

            for dep_idx in range(dependent_jobs + 1):
                # Build generation parameters dict for Script
                generation_params = {
                    "output_dir": output_dir,
                    "input_file": input_file,
                    "input_dir": input_dir,
                    "extra_arguments": extra_arguments,
                    "random_seed": seed,
                    "chunk_id": chunk_id,
                    "num_chunks": num_chunks,
                    "preprocess_cmd": preprocess_cmd,
                    "postprocess_cmd": postprocess_cmd,
                    "wandb_parameters": wandb_parameters if seed_idx == 0 else None,
                    "script": generation_module,
                    "requirements": generation_requirements,
                    # Multi-model specific fields
                    "server_addresses_prehosted": server_addresses_resolved,
                    "model_names": models_list,
                    "server_types": server_types_list,
                }

                # Create CommandGroup(s) using Script objects
                # For multi-model, this creates multiple CommandGroups (one per model + one for client)
                # For single-model, this creates a single CommandGroup
                job_groups = _create_job_unified(
                    models=models_list,
                    server_configs=[cfg.copy() if cfg else None for cfg in server_configs],
                    generation_params=generation_params,
                    cluster_config=cluster_config,
                    installation_command=installation_command,
                    with_sandbox=with_sandbox,
                    partition=partition,
                    keep_mounts_for_sandbox=keep_mounts_for_sandbox,
                    task_name=task_name,
                    log_dir=log_dir,
                    sbatch_kwargs=sbatch_kwargs,
                    sandbox_env_overrides=sandbox_env_overrides,
                )

                # Use unique internal job name for dependency tracking, but same task_name
                internal_job_name = f"{task_name}-dep{dep_idx}" if dep_idx > 0 else task_name

                # Build dependencies: first job in chain gets external dependencies, rest chain to previous
                if dep_idx == 0:
                    # First job: add run_after if no task_dependencies
                    job_deps = dependencies.copy() if dependencies else []
                    if not dependencies and run_after:
                        run_after_list = run_after if isinstance(run_after, list) else [run_after]
                        job_deps.extend(run_after_list)
                    job_deps = job_deps if job_deps else None
                else:
                    # Subsequent jobs in chain depend on previous job (use job object, not string)
                    job_deps = [prev_job]

                # For multi-group jobs, use "groups" key; for single-group, use "group" key
                job_spec = {
                    "name": internal_job_name,
                    "dependencies": job_deps,
                }
                if len(job_groups) > 1:
                    job_spec["groups"] = job_groups
                else:
                    job_spec["group"] = job_groups[0]

                jobs.append(job_spec)
                prev_job = job_spec  # Track for next iteration

                all_job_names.append(internal_job_name)

    # If no jobs to run, return early
    if not jobs:
        return None

    # Create and run pipeline
    pipeline = Pipeline(
        name=expname,
        cluster_config=cluster_config,
        jobs=jobs,
        reuse_code=reuse_code,
        reuse_code_exp=reuse_code_exp,
        skip_hf_home_check=skip_hf_home_check,
    )

    # TODO: remove after https://github.com/NVIDIA-NeMo/Skills/issues/578 is resolved as default will be single job
    sequential = True if cluster_config["executor"] in ["local", "none"] else False

    # Pass _reuse_exp to pipeline.run() to add jobs to existing experiment
    result = pipeline.run(dry_run=dry_run, _reuse_exp=_reuse_exp, sequential=sequential)
    return result


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
