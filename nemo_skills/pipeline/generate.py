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
from typing import List

import typer

import nemo_skills.pipeline.utils as pipeline_utils
from nemo_skills.dataset.utils import import_from_path
from nemo_skills.inference import GENERATION_MODULE_MAP, GenerationType
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.utils import (
    compute_chunk_ids,
    get_logger_name,
    setup_logging,
    str_ids_to_list,
    validate_wandb_project_name,
)

LOG = logging.getLogger(get_logger_name(__file__))

# TODO: add num_jobs here for consistency with eval?


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
    time_min: str = typer.Option(None, help="If specified, will use as a time-min slurm parameter"),
    eval_args: str = typer.Option(
        None, help="Specify if need to run nemo_skills/evaluation/evaluate_results.py on the generation outputs"
    ),
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
    skip_hf_home_check: bool = typer.Option(
        False,
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

    if os.sep in generation_module:
        generation_task = import_from_path(generation_module)
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
    has_tasks = False
    all_tasks = []
    if _task_dependencies is None:
        _task_dependencies = []
    with pipeline_utils.get_exp(expname, cluster_config, _reuse_exp) as exp:
        for seed_idx, (seed, chunk_ids) in enumerate(remaining_jobs.items()):
            if wandb_parameters:
                # no need for chunks as it will run after merging
                wandb_parameters["samples_file"] = pipeline_utils.get_chunked_rs_filename(
                    output_dir,
                    random_seed=seed,
                    chunk_id=None,
                )
            for chunk_id in chunk_ids:
                has_tasks = True
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
                cmd = pipeline_utils.get_generation_cmd(
                    input_file=input_file,
                    input_dir=input_dir,
                    random_seed=seed,
                    output_dir=output_dir,
                    extra_arguments=extra_arguments,
                    eval_args=eval_args,
                    chunk_id=chunk_id,
                    num_chunks=num_chunks,
                    preprocess_cmd=preprocess_cmd,
                    postprocess_cmd=postprocess_cmd,
                    wandb_parameters=wandb_parameters if seed_idx == 0 else None,
                    script=generation_module,
                )
                prev_tasks = _task_dependencies
                for _ in range(dependent_jobs + 1):
                    task_name = f"{expname}-rs{seed}" if seed is not None else expname
                    if chunk_id is not None:
                        task_name += f"-chunk{chunk_id}"
                    new_task = pipeline_utils.add_task(
                        exp,
                        cmd=pipeline_utils.wrap_python_path(cmd=cmd),
                        task_name=task_name,
                        log_dir=log_dir,
                        container=cluster_config["containers"]["nemo-skills"],
                        cluster_config=cluster_config,
                        partition=partition,
                        time_min=time_min,
                        server_config=server_config,
                        with_sandbox=with_sandbox,
                        keep_mounts_for_sandbox=keep_mounts_for_sandbox,
                        sandbox_port=None if get_random_port else 6000,
                        run_after=run_after,
                        reuse_code=reuse_code,
                        reuse_code_exp=reuse_code_exp,
                        task_dependencies=(
                            prev_tasks if cluster_config["executor"] == "slurm" else all_tasks + _task_dependencies
                        ),
                        get_server_command=generation_task.get_server_command_fn(),
                        slurm_kwargs={"exclusive": exclusive} if exclusive else None,
                        installation_command=installation_command,
                        skip_hf_home_check=skip_hf_home_check,
                    )
                    prev_tasks = [new_task]
                    all_tasks.append(new_task)
        if has_tasks and not _reuse_exp:  # if we are reusing an experiment, the tasks will run from there
            pipeline_utils.run_exp(exp, cluster_config, dry_run=dry_run)

    if _reuse_exp:
        return all_tasks
    else:
        if has_tasks:
            return exp
        return None


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
