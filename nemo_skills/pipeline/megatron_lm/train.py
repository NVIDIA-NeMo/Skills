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

import logging
from typing import List

import typer

from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.megatron_lm import megatron_lm_app
from nemo_skills.pipeline.utils import (
    add_task,
    check_if_mounted,
    check_mounts,
    get_cluster_config,
    get_exp,
    get_timeout_str,
    parse_sbatch_kwargs,
    resolve_mount_paths,
    run_exp,
)
from nemo_skills.utils import get_logger_name, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


def get_training_cmd(
    cluster_config,
    partition,
    entrypoint,
    init_cmd,
    megatron_model,
    tokenizer_model,
    per_split_data_path,
    expname,
    output_dir,
    disable_wandb,
    wandb_project,
    wandb_group,
    extra_arguments,
):
    """Generate Megatron-LM training command."""

    timeout_str = get_timeout_str(cluster_config, partition)
    # timeout_str is in NeMo-RL format: DD:HH:MM:SS (see get_timeout_str)
    days_str, hours_str, mins_str, secs_str = timeout_str.split(":")
    days, hours, mins, secs = int(days_str), int(hours_str), int(mins_str), int(secs_str)
    timeout_minutes = days * 24 * 60 + hours * 60 + mins
    # round up if there are leftover seconds
    if secs > 0:
        timeout_minutes += 1

    # Base command setup
    cmd = (
        f"echo 'Running init command' && "
        f"{init_cmd} && "
        "echo 'Starting Megatron-LM training' && "
        f"cd /opt/Megatron-LM && "
        f"python -u {entrypoint} "
        f"    --pretrained-checkpoint {megatron_model} "
        f"    --tokenizer-model {tokenizer_model} "
        f"    --per-split-data-args-path {per_split_data_path} "
        f"    --load {output_dir}/checkpoints "
        f"    --save {output_dir}/checkpoints "
        f"    --tensorboard-dir {output_dir}/tensorboard "
        f"    --data-cache-path {output_dir}/megatron-lm-data-cache "  # unused for sft
        f"    --exit-duration-in-mins {timeout_minutes} "
    )

    # Add wandb configuration if enabled
    if not disable_wandb:
        cmd += f"--wandb-project {wandb_project} --wandb-exp-name {expname} "
        if wandb_group:
            cmd += f"--wandb-group {wandb_group} "

    cmd += f" {extra_arguments} "

    return cmd


@megatron_lm_app.command(
    name="megatron-lm train", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
@typer_unpacker
def train_megatron_lm(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    output_dir: str = typer.Option(..., help="Where to put results"),
    expname: str = typer.Option("megatron-lm-train", help="Experiment name"),
    entrypoint: str = typer.Option(..., help="Entrypoint script name, e.g. pretrain_gpt.py or pretrain_mamba.py"),
    init_cmd: str = typer.Option(
        "",
        help="Initialization command to run before the main command. "
        "Useful to include a large list of environment variables.",
    ),
    megatron_model: str = typer.Option(..., help="Path to the Megatron model checkpoint"),
    tokenizer_model: str = typer.Option(..., help="Path to the tokenizer model"),
    per_split_data_path: str = typer.Option(
        ..., help="Path to the json file containing information about per-split training data"
    ),
    num_nodes: int = typer.Option(1, help="Number of nodes"),
    num_gpus: int = typer.Option(..., help="Number of GPUs per node"),
    num_training_jobs: int = typer.Option(1, help="Number of training jobs"),
    wandb_project: str = typer.Option("nemo-skills", help="Weights & Biases project name"),
    wandb_group: str = typer.Option(None, help="Weights & Biases group name."),
    disable_wandb: bool = typer.Option(False, help="Disable wandb logging"),
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
    log_dir: str = typer.Option(
        None,
        help="Can specify a custom location for slurm logs. "
        "If not specified, will be inside `ssh_tunnel.job_dir` part of your cluster config.",
    ),
    exclusive: bool = typer.Option(
        True,
        "--not_exclusive",
        help="If --not_exclusive is used, will NOT use --exclusive flag for slurm",
    ),
    mount_paths: str = typer.Option(None, help="Comma separated list of paths to mount on the remote machine"),
    check_mounted_paths: bool = typer.Option(False, help="Check if mounted paths are available on the remote machine"),
    skip_hf_home_check: bool | None = typer.Option(
        None,
        help="If True, skip checking that HF_HOME env var is defined in the cluster config.",
    ),
    installation_command: str | None = typer.Option(
        None,
        help="An installation command to run before main job. Only affects main task (not server or sandbox). "
        "You can use an arbitrary command here and we will run it on a single rank for each node. "
        "E.g. 'pip install my_package'",
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
    """Runs Megatron-LM SFT training.

    All extra arguments are passed directly to the Megatron-LM training script.
    """
    setup_logging(disable_hydra_logs=False, use_rich=True)
    extra_arguments = f"{' '.join(ctx.args)}"
    LOG.info("Starting Megatron-LM SFT training job")
    LOG.info("Extra arguments that will be passed to the underlying script: %s", extra_arguments)

    cluster_config = get_cluster_config(cluster, config_dir)
    cluster_config = resolve_mount_paths(cluster_config, mount_paths)

    if log_dir is None:
        log_dir = output_dir

    output_dir, megatron_model, per_split_data_path, log_dir = check_mounts(
        cluster_config,
        log_dir=log_dir,
        mount_map={output_dir: None, megatron_model: None, per_split_data_path: None},
        check_mounted_paths=check_mounted_paths,
    )

    if tokenizer_model.startswith("/"):
        check_if_mounted(cluster_config, tokenizer_model)

    train_cmd = get_training_cmd(
        cluster_config=cluster_config,
        partition=partition,
        entrypoint=entrypoint,
        init_cmd=init_cmd,
        megatron_model=megatron_model,
        tokenizer_model=tokenizer_model,
        per_split_data_path=per_split_data_path,
        expname=expname,
        output_dir=output_dir,
        disable_wandb=disable_wandb,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        extra_arguments=extra_arguments,
    )

    with get_exp(expname, cluster_config, _reuse_exp) as exp:
        prev_task = _task_dependencies
        for job_id in range(num_training_jobs):
            prev_task = add_task(
                exp,
                cmd=train_cmd,
                task_name=f"{expname}-{job_id}",
                log_dir=f"{log_dir}/training-logs",
                container=cluster_config["containers"]["megatron"],
                num_tasks=num_gpus,
                num_gpus=num_gpus,
                num_nodes=num_nodes,
                cluster_config=cluster_config,
                partition=partition,
                run_after=run_after,
                reuse_code=reuse_code,
                reuse_code_exp=reuse_code_exp,
                task_dependencies=[prev_task] if prev_task is not None else None,
                sbatch_kwargs=parse_sbatch_kwargs(sbatch_kwargs, exclusive=exclusive, qos=qos, time_min=time_min),
                installation_command=installation_command,
                skip_hf_home_check=skip_hf_home_check,
            )
        run_exp(exp, cluster_config, sequential=False, dry_run=dry_run)

    if _reuse_exp:
        return [prev_task]
    return exp


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
