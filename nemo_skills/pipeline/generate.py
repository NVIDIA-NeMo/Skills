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
from typing import Callable, Dict, List, Optional

import typer

import nemo_skills.pipeline.utils as pipeline_utils
from nemo_skills.dataset.utils import import_from_path
from nemo_skills.inference import GENERATION_MODULE_MAP, GenerationType
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils.commands import sandbox_command
from nemo_skills.pipeline.utils.declarative import Command, CommandGroup, HardwareConfig, Pipeline
from nemo_skills.pipeline.utils.server import get_free_port
from nemo_skills.utils import (
    compute_chunk_ids,
    get_logger_name,
    setup_logging,
    str_ids_to_list,
    validate_wandb_project_name,
)

LOG = logging.getLogger(get_logger_name(__file__))

# TODO: add num_jobs here for consistency with eval?


def _create_commandgroup_from_config(
    generation_cmd: str,
    server_config: Optional[Dict],
    with_sandbox: bool,
    sandbox_port: Optional[int],
    cluster_config: Dict,
    installation_command: Optional[str],
    get_server_command_fn: Callable,
    partition: Optional[str],
    qos: Optional[str],
    time_min: Optional[str],
    exclusive: bool,
    keep_mounts_for_sandbox: bool,
    task_name: str,
    log_dir: str,
) -> CommandGroup:
    """Create a CommandGroup from server_config.

    Component ordering:
    1. Server (if server_config provided)
    2. Client command
    3. Sandbox (if with_sandbox=True)
    """

    components = []

    # 1. Add server if server_config is provided
    if server_config is not None and int(server_config["num_gpus"]) > 0:
        server_type = server_config["server_type"]
        # Get container from server_config if provided, otherwise fall back to cluster config
        if "container" in server_config:
            server_container = server_config.pop("container")
        else:
            server_container = cluster_config["containers"][server_type]

        # Call server command builder directly with cluster_config
        cmd, num_tasks = get_server_command_fn(**server_config, cluster_config=cluster_config)

        # Create metadata dict
        metadata = {
            "num_tasks": num_tasks,
            "gpus": server_config["num_gpus"],
            "nodes": server_config["num_nodes"],
            "log_prefix": "server",
        }

        server_cmd = Command(
            command=cmd,
            container=server_container,
            gpus=server_config["num_gpus"],
            nodes=server_config["num_nodes"],
            name=task_name,
            metadata=metadata,
        )
        components.append(server_cmd)

    # 2. Add main generation command
    # Note: General cluster config env vars are automatically added by get_env_variables() in get_executor()
    client_env = {}
    if with_sandbox and sandbox_port is not None:
        client_env["NEMO_SKILLS_SANDBOX_PORT"] = str(sandbox_port)

    client_cmd = Command(
        command=generation_cmd,
        container=cluster_config["containers"]["nemo-skills"],
        name=task_name,
        installation_command=installation_command,
        metadata={
            "log_prefix": "main",
            "environment": client_env,
        },
    )
    components.append(client_cmd)

    # 3. Add sandbox if requested
    if with_sandbox:
        # Call sandbox command builder directly with cluster_config
        cmd, metadata = sandbox_command(cluster_config=cluster_config, port=sandbox_port)
        metadata["log_prefix"] = "sandbox"

        sandbox_cmd = Command(
            command=cmd,
            container=cluster_config["containers"]["sandbox"],
            name=task_name,
            metadata=metadata,
        )

        components.append(sandbox_cmd)

    # Find maximum GPUs/nodes needed by any component for the HardwareConfig
    # The job-level resource request must be the maximum across all components
    max_gpus = max((comp.gpus or 0) for comp in components)
    max_nodes = max((comp.nodes or 1) for comp in components)

    return CommandGroup(
        commands=components,
        hardware=HardwareConfig(
            partition=partition,
            qos=qos,
            time_min=time_min,
            exclusive=exclusive,
            num_gpus=max_gpus,
            num_nodes=max_nodes,
        ),
        name=task_name,
        log_dir=log_dir,
    )


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
    model: str = typer.Option(None, help="Path to the model or model name in API"),
    server_address: str = typer.Option(
        None, help="Use ip:port for self-hosted models or the API url if using model providers"
    ),
    server_type: pipeline_utils.SupportedServers = typer.Option(..., help="Type of server to use"),
    server_gpus: int = typer.Option(None, help="Number of GPUs to use if hosting the model"),
    server_nodes: int = typer.Option(1, help="Number of nodes required for hosting LLM server"),
    server_args: str = typer.Option("", help="Any extra arguments to pass to the server"),
    server_entrypoint: str = typer.Option(
        None,
        help="Path to the entrypoint of the server. "
        "If not specified, will use the default entrypoint for the server type.",
    ),
    server_container: str = typer.Option(
        None, help="Override container image for the hosted server (if server_gpus is set)"
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
    exclusive: bool = typer.Option(False, help="If set will add exclusive flag to the slurm job."),
    rerun_done: bool = typer.Option(
        False, help="If True, will re-run jobs even if a corresponding '.done' file already exists"
    ),
    with_sandbox: bool = typer.Option(False, help="If True, will start a sandbox container alongside this job"),
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
    _reuse_exp: str = typer.Option(None, help="Internal option to reuse an experiment object.", hidden=True),
    _task_dependencies: List[str] = typer.Option(
        None, help="Internal option to specify task dependencies.", hidden=True
    ),
):
    """Generate LLM completions for a given input file.

    Run `python -m nemo_skills.inference.generate --help` for other supported arguments
    (need to be prefixed with ++, since we use Hydra for that script).
    """
    setup_logging(disable_hydra_logs=False, use_rich=True)
    extra_arguments = f"{' '.join(ctx.args)}"
    LOG.info("Starting generation job")
    LOG.info("Extra arguments that will be passed to the underlying script: %s", extra_arguments)

    try:
        server_type = server_type.value
    except AttributeError:
        pass

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

    get_random_port = pipeline_utils.should_get_random_port(server_gpus, exclusive)

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

    original_server_address = server_address

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
            # Configure client (same as before)
            server_config, server_address, extra_arguments = pipeline_utils.configure_client(
                model=model,
                server_type=server_type,
                server_address=original_server_address,
                server_gpus=server_gpus,
                server_nodes=server_nodes,
                server_args=server_args,
                server_entrypoint=server_entrypoint,
                server_container=server_container,
                extra_arguments=extra_arguments_original,
                get_random_port=get_random_port,
            )

            # Build generation command (same as before)
            cmd = pipeline_utils.get_generation_cmd(
                input_file=input_file,
                input_dir=input_dir,
                random_seed=seed,
                output_dir=output_dir,
                extra_arguments=extra_arguments,
                chunk_id=chunk_id,
                num_chunks=num_chunks,
                preprocess_cmd=preprocess_cmd,
                postprocess_cmd=postprocess_cmd,
                wandb_parameters=wandb_parameters if seed_idx == 0 else None,
                script=generation_module,
            )
            cmd = pipeline_utils.wrap_python_path(cmd=cmd)

            # Base task name (shared across all dependent jobs in the chain)
            task_name = f"{expname}-rs{seed}" if seed is not None else expname
            if chunk_id is not None:
                task_name += f"-chunk{chunk_id}"

            # Handle dependent_jobs chain
            dependencies = _task_dependencies.copy() if _task_dependencies else []
            prev_job = None

            for dep_idx in range(dependent_jobs + 1):
                # Allocate sandbox port if needed
                # This must be done BEFORE creating CommandGroup so client knows the port
                if with_sandbox:
                    current_sandbox_port = get_free_port(strategy="random") if get_random_port else 6000
                else:
                    current_sandbox_port = None

                # Create CommandGroup for this task
                cmd_group = _create_commandgroup_from_config(
                    generation_cmd=cmd,
                    server_config=server_config.copy() if server_config else None,
                    with_sandbox=with_sandbox,
                    sandbox_port=current_sandbox_port,
                    cluster_config=cluster_config,
                    installation_command=installation_command,
                    get_server_command_fn=generation_task.get_server_command_fn(),
                    partition=partition,
                    qos=qos,
                    time_min=time_min,
                    exclusive=exclusive,
                    keep_mounts_for_sandbox=keep_mounts_for_sandbox,
                    task_name=task_name,
                    log_dir=log_dir,
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

                job_spec = {
                    "name": internal_job_name,
                    "group": cmd_group,
                    "dependencies": job_deps,
                }
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
