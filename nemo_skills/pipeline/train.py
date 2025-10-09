# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List

import typer

from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils import (
    add_task,
    check_mounts,
    get_cluster_config,
    get_exp,
    get_mounted_path,
    get_timeout_str,
    resolve_mount_paths,
    run_exp,
)
from nemo_skills.utils import get_logger_name, setup_logging, validate_wandb_project_name

LOG = logging.getLogger(get_logger_name(__file__))


class TrainingAlgo(str, Enum):
    sft = "sft"
    dpo = "dpo"
    rm = "rm"


@dataclass
class TrainingParams:
    training_script: str
    config_params: str
    nemo_model: str
    output_dir: str
    training_data: str
    validation_data: str
    num_gpus: int
    num_nodes: int
    expname: str
    training_algo: TrainingAlgo
    disable_wandb: bool
    wandb_project: str
    wandb_group: str
    timeout: str
    extra_arguments: str = ""
    logging_params: str = ""

    def __post_init__(self):
        self.extra_arguments = get_extra_arguments[self.training_algo](self)


def get_cmd(params: TrainingParams) -> str:
    cmd = (
        f"export HYDRA_FULL_ERROR=1 && "
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"export CUDA_DEVICE_MAX_CONNECTIONS=1 && "
        f"export TEMPDIR=/dev/shm/checkpoints_$SLURM_JOB_ID && "
        f"mkdir -p $TEMPDIR && "
        f"chmod 777 $TEMPDIR && "
        f"cd /nemo_run/code && "
        f"echo 'Starting training' && "
        f"{params.training_script} "
        f"    {params.config_params}"
        f"    ++model.tensor_model_parallel_size={params.num_gpus} "
        f"    trainer.devices={params.num_gpus} "
        f"    trainer.num_nodes={params.num_nodes} "
        f"    {params.logging_params} "
        f"    exp_manager.name={params.expname} "
        f"    exp_manager.explicit_log_dir={params.output_dir}/training "
        f"    exp_manager.exp_dir={params.output_dir}/training "
        f"    ++exp_manager.max_time_per_run={params.timeout} "
        f"    {params.extra_arguments} "
    )
    return cmd


configs = {
    TrainingAlgo.sft: "sft_config",
    TrainingAlgo.dpo: "dpo_config",
    TrainingAlgo.rm: "rm_config",
}


def rl_extra_args_fn(params):
    return (
        f" ++model.data.data_prefix.train='[{params.training_data}]' "
        f" ++model.data.data_prefix.validation='[{params.validation_data}]' "
        f" ++model.data.data_prefix.test='[{params.validation_data}]' "
        f" pretrained_checkpoint.restore_from_path={params.nemo_model} " + params.extra_arguments
    )


get_extra_arguments: dict[TrainingAlgo, Callable[[TrainingParams], str]] = {
    TrainingAlgo.sft: lambda params: (
        f" ++model.data.train_ds.file_path='{params.training_data}' "
        f" ++model.data.train_ds.index_mapping_dir='{os.path.dirname(os.path.abspath(params.training_data))}' "
        f" ++model.data.validation_ds.file_path='{params.validation_data}' "
        f" ++model.data.validation_ds.index_mapping_dir='{os.path.dirname(os.path.abspath(params.validation_data))}' "
        f" model.restore_from_path={params.nemo_model} " + params.extra_arguments
    ),
    TrainingAlgo.dpo: rl_extra_args_fn,
    TrainingAlgo.rm: rl_extra_args_fn,
}


def get_training_cmd(
    cluster_config,
    partition,
    config_name,
    config_path,
    nemo_model,
    output_dir,
    training_data,
    validation_data,
    num_gpus,
    num_nodes,
    expname,
    training_algo,
    disable_wandb,
    wandb_project,
    wandb_group,
    extra_arguments,
):
    if validation_data is None:
        validation_data = training_data

    timeout = get_timeout_str(cluster_config, partition)

    logging_params = get_logging_params(expname, disable_wandb, wandb_project, wandb_group)

    if config_name is None:
        config_name = configs[training_algo]
    config_params = f"--config-name={config_name} --config-path={config_path} "

    training_script = f"python -m nemo_skills.training.start_{training_algo}"

    training_params = TrainingParams(
        training_script=training_script,
        config_params=config_params,
        nemo_model=nemo_model,
        output_dir=output_dir,
        training_data=training_data,
        validation_data=validation_data,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        expname=expname,
        training_algo=training_algo,
        disable_wandb=disable_wandb,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        timeout=timeout,
        extra_arguments=extra_arguments,
        logging_params=logging_params,
    )

    return get_cmd(training_params), training_params


def get_logging_params(expname, disable_wandb, wandb_project, wandb_group):
    if not disable_wandb:
        if os.getenv("WANDB_API_KEY") is None:
            raise ValueError("WANDB_API_KEY is not set. Use --disable_wandb to disable wandb logging")
        wandb_id = expname + ("-" + wandb_group if wandb_group else "") + "-" + wandb_project
        logging_params = (
            f"exp_manager.create_wandb_logger=True "
            f"exp_manager.wandb_logger_kwargs.name={expname} "
            f"exp_manager.wandb_logger_kwargs.project={wandb_project} "
            f"+exp_manager.wandb_logger_kwargs.id={wandb_id} "
            f"+exp_manager.wandb_logger_kwargs.resume=True "
        )
        if wandb_group:
            logging_params += f"++exp_manager.wandb_logger_kwargs.group={wandb_group} "

        validate_wandb_project_name(
            wandb_project=wandb_project,
            wandb_name=expname,
            wandb_group=wandb_group,
            wandb_id=wandb_id,
        )
    else:
        logging_params = "exp_manager.create_wandb_logger=False +exp_manager.create_tensorboard_logger=True"
    return logging_params


def get_checkpoint_cmd(nemo_model, output_dir, final_nemo_path, average_steps):
    if average_steps is not None:
        average_steps = f"--steps {' '.join(average_steps.split(','))} " if average_steps != "all" else ""
        entrypoint = "nemo_skills.training.average_checkpoints"
        name = "model" + ("-".join(average_steps[len("--steps ") :].split()) if average_steps else "") + "-averaged"
    else:
        entrypoint = "nemo_skills.training.copy_checkpoint"
        name = "model-last"

    average_steps_arg = average_steps if average_steps else ""

    cmd = (
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"cd /nemo_run/code && "
        f"python -m {entrypoint} "
        f"    --untarred_nemo_dir {nemo_model} "
        f"    --name_prefix=model "
        f"    --checkpoint_dir={output_dir}/training/checkpoints {average_steps_arg} && "
        f"mkdir -p {os.path.dirname(final_nemo_path)} && "
        f"mv {output_dir}/training/checkpoints/{name} {final_nemo_path} "
    )
    return cmd


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def train(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    output_dir: str = typer.Option(..., help="Where to put results"),
    final_nemo_path: str = typer.Option(None, help="Where to put the final checkpoint"),
    expname: str = typer.Option(..., help="Experiment name"),
    nemo_model: str = typer.Option(..., help="Path to the NeMo model"),
    training_data: str = typer.Option(None, help="Path to the training data"),
    validation_data: str = typer.Option(None, help="Path to the validation data"),
    num_nodes: int = typer.Option(1, help="Number of nodes"),
    num_gpus: int = typer.Option(..., help="Number of GPUs"),
    num_training_jobs: int = typer.Option(1, help="Number of training jobs"),
    training_algo: TrainingAlgo = typer.Option(TrainingAlgo.sft, help="Training algorithm"),
    config_name: str = typer.Option(None, help="Config name"),
    config_path: str = typer.Option("/nemo_run/code/nemo_skills/training/", help="Config path"),
    wandb_group: str = typer.Option(None, help="Weights & Biases group name."),
    wandb_project: str = typer.Option("nemo-skills", help="Weights & Biases project name"),
    disable_wandb: bool = typer.Option(False, help="Disable wandb logging"),
    with_sandbox: bool = typer.Option(False, help="If sandbox is required for code generation"),
    keep_mounts_for_sandbox: bool = typer.Option(
        False,
        help="If True, will keep the mounts for the sandbox container. Note that, it is risky given that sandbox executes LLM commands and could potentially lead to data loss. So, we advise not to use this unless absolutely necessary.",
    ),
    partition: str = typer.Option(None, help="Specify partition for jobs"),
    time_min: str = typer.Option(None, help="If specified, will use as a time-min slurm parameter"),
    average_steps: str = typer.Option(
        "all",
        help="List of commas separated checkpoint steps to average. E.g 1000,5000. If None, skip prepare eval stage.",
    ),
    save_last_ckpt: bool = typer.Option(
        False,
        help="If True, will save the final nemo checkpoint to final_nemo_path. "
        "average_steps has to be disabled to use this.",
    ),
    mount_paths: str = typer.Option(None, help="Comma separated list of paths to mount on the remote machine"),
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
    log_dir: str = typer.Option(None, help="Can specify a custom location for slurm logs. "),
    exclusive: bool = typer.Option(False, help="If set will add exclusive flag to the slurm job."),
    check_mounted_paths: bool = typer.Option(False, help="Check if mounted paths are available on the remote machine"),
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
    """Train (SFT or DPO) an LLM model.

    All extra arguments are passed directly to the training script
    (need to be prefixed with ++, since NeMo uses Hydra).
    """
    setup_logging(disable_hydra_logs=False, use_rich=True)
    extra_arguments = f"{' '.join(ctx.args)}"
    LOG.info("Starting training job")
    LOG.info("Extra arguments that will be passed to the underlying script: %s", extra_arguments)

    try:
        training_algo = training_algo.value
    except AttributeError:
        pass

    cluster_config = get_cluster_config(cluster, config_dir)
    cluster_config = resolve_mount_paths(cluster_config, mount_paths)

    if log_dir is None:
        log_dir = f"{output_dir}"

    nemo_model, output_dir, log_dir = check_mounts(
        cluster_config,
        log_dir=log_dir,
        mount_map={nemo_model: None, output_dir: None},
        check_mounted_paths=check_mounted_paths,
    )

    if num_training_jobs > 0:
        if training_data is None:
            raise ValueError("training_data is required when num_training_jobs > 0")
        training_data = get_mounted_path(cluster_config, training_data)

    if not final_nemo_path:
        final_nemo_path = f"{output_dir}/model-averaged-nemo"
    final_nemo_path = get_mounted_path(cluster_config, final_nemo_path)

    if validation_data:
        validation_data = get_mounted_path(cluster_config, validation_data)

    if " " in str(average_steps):
        raise ValueError("average steps should be separated with commas")

    if average_steps and save_last_ckpt:
        raise ValueError("cannot enable average_steps and save_last_ckpt together.")

    train_cmd, training_params = get_training_cmd(
        cluster_config=cluster_config,
        partition=partition,
        config_name=config_name,
        config_path=config_path,
        nemo_model=nemo_model,
        output_dir=output_dir,
        training_data=training_data,
        validation_data=validation_data,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        expname=expname,
        training_algo=training_algo,
        disable_wandb=disable_wandb,
        wandb_project=wandb_project,
        wandb_group=wandb_group,
        extra_arguments=extra_arguments,
    )
    container = cluster_config["containers"]["nemo"]
    num_tasks = num_gpus if cluster_config["executor"] == "slurm" else 1

    with get_exp(expname, cluster_config, _reuse_exp) as exp:
        prev_task = _task_dependencies
        for job_id in range(num_training_jobs):
            prev_task = add_task(
                exp,
                cmd=train_cmd,
                task_name=f"{expname}-{training_algo}-{job_id}",
                log_dir=f"{log_dir}/training-logs",
                container=container,
                num_gpus=num_gpus,
                num_nodes=num_nodes,
                num_tasks=num_tasks,
                cluster_config=cluster_config,
                partition=partition,
                time_min=time_min,
                with_sandbox=with_sandbox,
                keep_mounts_for_sandbox=keep_mounts_for_sandbox,
                run_after=run_after,
                reuse_code=reuse_code,
                reuse_code_exp=reuse_code_exp,
                task_dependencies=[prev_task] if prev_task is not None else None,
                slurm_kwargs={"exclusive": exclusive} if exclusive else None,
                installation_command=installation_command,
                skip_hf_home_check=skip_hf_home_check,
            )

        if average_steps or save_last_ckpt:
            cmd = get_checkpoint_cmd(
                nemo_model=nemo_model,
                output_dir=output_dir,
                final_nemo_path=final_nemo_path,
                average_steps=average_steps,
            )

            prev_task = add_task(
                exp,
                cmd=cmd,
                task_name=f"{expname}-prepare-eval",
                log_dir=f"{log_dir}/prepare-eval-logs",
                container=cluster_config["containers"]["nemo"],
                cluster_config=cluster_config,
                partition=partition,
                time_min=time_min,
                num_nodes=1,
                num_tasks=1,
                num_gpus=num_gpus,
                run_after=run_after,
                reuse_code=reuse_code,
                reuse_code_exp=reuse_code_exp,
                task_dependencies=[prev_task] if prev_task is not None else None,
                slurm_kwargs={"exclusive": exclusive} if exclusive else None,
                installation_command=installation_command,
                skip_hf_home_check=skip_hf_home_check,
            )

        # explicitly setting sequential to False since we set dependencies directly
        run_exp(exp, cluster_config, sequential=False, dry_run=dry_run)

    if _reuse_exp:
        return [prev_task]
    return exp


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    app()
