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

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import typer

import nemo_skills.pipeline.utils as pipeline_utils
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils import parse_kwargs
from nemo_skills.pipeline.utils.server import get_free_port
from nemo_skills.pipeline.verl import verl_app
from nemo_skills.utils import (
    get_logger_name,
    setup_logging,
    validate_wandb_project_name,
)

LOG = logging.getLogger(get_logger_name(__file__))


@dataclass
class PPOVerlTask:
    model: str
    output_dir: str
    prompt_data: str
    eval_data: str
    num_gpus: int
    num_nodes: int
    expname: str
    disable_wandb: bool
    wandb_project: str
    timeout: str
    extra_arguments: str = ""
    logging_params: str = ""
    script_module: str = "verl.trainer.main_ppo"
    verl_config_dir: str = None
    verl_config_name: str = None

    def get_ray_launch_cmd(self):
        cmd = "ray job submit --address='http://127.0.0.1:8265' -- "
        return cmd

    def format_train_args(self):
        verl_config = (
            ""
            if ((self.verl_config_dir is None) and (self.verl_config_name is None))
            else f" --config-path {self.verl_config_dir} --config-name {self.verl_config_name} "
        )
        if verl_config == "":
            cmd = (
                "   algorithm.adv_estimator=grpo "
                "   data.train_batch_size=128 "
                "   data.val_batch_size=512 "
                "   data.max_prompt_length=1024 "
                "   data.max_response_length=8192 "
                "   actor_rollout_ref.actor.optim.lr=1e-6 "
                "   actor_rollout_ref.model.use_remove_padding=True "
                "   actor_rollout_ref.actor.ppo_mini_batch_size=64 "
                "   actor_rollout_ref.actor.use_dynamic_bsz=True "
                "   actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 "
                "   actor_rollout_ref.actor.use_kl_loss=True "
                "   actor_rollout_ref.actor.kl_loss_coef=0.0 "
                "   actor_rollout_ref.actor.kl_loss_type=low_var_kl "
                "   actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 "
                "   actor_rollout_ref.model.enable_gradient_checkpointing=True "
                "   actor_rollout_ref.actor.fsdp_config.param_offload=False "
                "   +actor_rollout_ref.actor.fsdp_config.grad_offload=False "
                "   actor_rollout_ref.actor.fsdp_config.optimizer_offload=False "
                "   actor_rollout_ref.rollout.tensor_model_parallel_size=1 "
                "   actor_rollout_ref.rollout.name=vllm "
                "   actor_rollout_ref.rollout.temperature=0.6 "
                "   actor_rollout_ref.rollout.gpu_memory_utilization=0.85 "
                "   actor_rollout_ref.rollout.enforce_eager=False "
                "   actor_rollout_ref.rollout.free_cache_engine=False "
                "   actor_rollout_ref.rollout.n=8 "
                "   actor_rollout_ref.ref.fsdp_config.param_offload=True "
                "   algorithm.kl_ctrl.kl_coef=0 "
                "   trainer.critic_warmup=0 "
                "   ++trainer.val_before_train=False "
                "   +trainer.val_generations_to_log_to_wandb=1 "
                "   trainer.save_freq=20 "
                "   trainer.test_freq=20 "
                "   trainer.default_hdfs_dir=null "
                "   trainer.total_epochs=30 "
                ""
            )
        else:
            cmd = f"  {verl_config} "

        cmd += (
            f"   actor_rollout_ref.model.path={self.model} "
            f"   trainer.default_local_dir={self.output_dir}/checkpoints "
            f"   trainer.n_gpus_per_node={self.num_gpus} "
            f"   trainer.nnodes={self.num_nodes} "
            f"  +trainer.timeout={self.timeout} "
        )

        return cmd

    def format_data_args(self):
        cmd = f"   data.train_files='{self.prompt_data}' "
        if self.eval_data:
            cmd = f"{cmd} data.val_files='{self.eval_data}' "
        else:
            cmd = f"{cmd} +trainer.run_validation=False "

        return cmd

    def format_wandb_args(self, disable_wandb, wandb_project, expname):
        cmd = f" trainer.project_name='{wandb_project}'  trainer.experiment_name='{expname}' "

        if disable_wandb:
            cmd = f"{cmd} trainer.logger=['console'] "
        else:
            cmd = f"{cmd} trainer.logger=['console','wandb'] "
            validate_wandb_project_name(
                wandb_project=wandb_project,
                wandb_name=expname,
            )

        return cmd

    def get_preamble_cmd(self):
        cmd = " echo 'No preamble command to execute, skipping...' "
        return cmd

    def get_script_module(self):
        return self.script_module

    def get_job_cmd(self):
        ray_job_cmd = self.get_ray_launch_cmd()
        ray_job_cmd = (
            f"echo 'Starting training' && "
            f"{ray_job_cmd} python3 -m {self.get_script_module()} "
            f"  {self.format_train_args()} "
            f"  {self.format_data_args()} "
            f"  {self.logging_params} "
            f"  {self.extra_arguments} "
        )
        return ray_job_cmd

    def get_cmd(self):
        self.logging_params = self.format_wandb_args(self.disable_wandb, self.wandb_project, self.expname)
        preamble_cmd = self.get_preamble_cmd()

        cmd = (
            f"export HYDRA_FULL_ERROR=1 && "
            f"export SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK=True && "
            f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
            f"export ROCR_VISIBLE_DEVICES= && "
            f"cd /nemo_run/code && "
            f"{preamble_cmd} && "
        )

        ray_job_cmd = self.get_job_cmd()
        ray_server_cmd = pipeline_utils.get_ray_server_cmd(ray_job_cmd)

        cmd = f"{cmd} {ray_server_cmd} "
        return cmd


def get_training_cmd(
    cluster_config,
    task: Optional[PPOVerlTask],
    partition,
    hf_model,
    output_dir,
    prompt_data,
    eval_data,
    num_gpus,
    num_nodes,
    expname,
    disable_wandb,
    wandb_project,
    extra_arguments,
    script_module="verl.trainer.main_ppo",
    verl_config_dir=None,
    verl_config_name=None,
):
    # TODO: use those
    timeout = pipeline_utils.get_timeout_str(cluster_config, partition)

    if task is None:
        task = PPOVerlTask(
            model=hf_model,
            output_dir=output_dir,
            prompt_data=prompt_data,
            eval_data=eval_data,
            num_gpus=num_gpus,
            num_nodes=num_nodes,
            expname=expname,
            disable_wandb=disable_wandb,
            wandb_project=wandb_project,
            timeout=timeout,
            extra_arguments=extra_arguments,
            logging_params="",  # Updated later
            script_module=script_module,
            verl_config_dir=verl_config_dir,
            verl_config_name=verl_config_name,
        )

    else:
        task.timeout = timeout
        task.extra_arguments = extra_arguments

    return task.get_cmd()


class SupportedServers(str, Enum):
    trtllm = "trtllm"
    vllm = "vllm"
    openai = "openai"
    sglang = "sglang"


@verl_app.command(name="ppo", context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def ppo_verl(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    output_dir: str = typer.Option(..., help="Where to put results"),
    expname: str = typer.Option("openrlhf-ppo", help="Nemo run experiment name"),
    hf_model: str = typer.Option(..., help="Path to the HF model"),
    prompt_data: str = typer.Option(None, help="Path to the prompt data"),
    eval_data: str = typer.Option(None, help="Path to the eval data"),
    num_nodes: int = typer.Option(1, help="Number of nodes"),
    num_gpus: int = typer.Option(..., help="Number of GPUs per node"),
    dependent_jobs: int = typer.Option(0, help="Number of dependent jobs"),
    server_model: str = typer.Option(None, help="Path to the model or model name in API"),
    server_address: str = typer.Option(
        None, help="Use ip:port for self-hosted models or the API url if using model providers"
    ),
    server_type: SupportedServers = typer.Option(None, help="Type of server to use"),
    server_gpus: int = typer.Option(None, help="Number of GPUs to use if hosting the model"),
    server_nodes: int = typer.Option(1, help="Number of nodes required for hosting LLM server"),
    server_args: str = typer.Option("", help="Any extra arguments to pass to the server"),
    wandb_project: str = typer.Option("nemo-skills", help="Weights & Biases project name"),
    disable_wandb: bool = typer.Option(False, help="Disable wandb logging"),
    final_ckpt_path: str = typer.Option(None, help="Where to put the final checkpoint"),
    convert_last_ckpt_to_hf: bool = typer.Option(
        False,
        help="If True, will convert the final checkpoint to hf format and place in final_ckpt_path (or output_dir/final_hf_checkpoint if not specified) ",
    ),
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
    with_sandbox: bool = typer.Option(
        False,
        help="If True, will use the sandbox to run the training job",
    ),
    script_module: str = typer.Option("verl.trainer.main_ppo", help="The script module to run. "),
    verl_config_dir: str = typer.Option(None, help="The directory containing the Verl config files. "),
    verl_config_name: str = typer.Option(None, help="The name of the Verl config file to use. "),
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
    """Runs Verl PPO training (verl.trainer.main_ppo)"""
    setup_logging(disable_hydra_logs=False, use_rich=True)
    extra_arguments = f"{' '.join(ctx.args)}"
    LOG.info("Starting training job")
    LOG.info("Extra arguments that will be passed to the underlying script: %s", extra_arguments)

    cluster_config = pipeline_utils.get_cluster_config(cluster, config_dir)
    pipeline_utils.check_if_mounted(cluster_config, output_dir)
    pipeline_utils.check_if_mounted(cluster_config, hf_model)
    if log_dir:
        pipeline_utils.check_if_mounted(cluster_config, log_dir)
    else:
        log_dir = output_dir

    if not final_ckpt_path:
        final_ckpt_path = f"{output_dir}/final_hf_checkpoint"
    pipeline_utils.check_if_mounted(cluster_config, final_ckpt_path)

    if dependent_jobs >= 0:
        if prompt_data is None:
            raise ValueError("prompt_data is required when ndependent_jobs > 0")
        if prompt_data.startswith("/"):  # could ask to download from HF
            pipeline_utils.check_if_mounted(cluster_config, prompt_data)

    # Check if custom PPOVerlTask is provided via ctx.obj['ppo_task'], use that if available
    if hasattr(ctx, "obj") and ctx.obj is not None and isinstance(ctx.obj, dict) and "ppo_task" in ctx.obj:
        ppo_task = ctx.obj["ppo_task"]  # type: type(PPOVerlTask)
        assert isinstance(ppo_task, PPOVerlTask), "`ppo_task` must be a subclass of PPOVerlTask"
    else:
        ppo_task = None

    train_cmd = get_training_cmd(
        cluster_config=cluster_config,
        task=ppo_task,
        partition=partition,
        hf_model=hf_model,
        output_dir=output_dir,
        prompt_data=prompt_data,
        eval_data=eval_data,
        num_gpus=num_gpus,
        num_nodes=num_nodes,
        expname=expname,
        disable_wandb=disable_wandb,
        wandb_project=wandb_project,
        extra_arguments=extra_arguments,
        script_module=script_module,
        verl_config_dir=verl_config_dir,
        verl_config_name=verl_config_name,
    )

    server_config = None
    if server_type is not None:
        get_random_port = pipeline_utils.should_get_random_port(server_gpus, exclusive)
        if server_address is None:  # we need to host the model
            assert server_gpus is not None, "Need to specify server_gpus if hosting the model"
            server_port = get_free_port(strategy="random") if get_random_port else 5000
            server_address = f"localhost:{server_port}"

            server_config = {
                "model_path": server_model,
                "server_type": server_type,
                "num_gpus": server_gpus,
                "num_nodes": server_nodes,
                "server_args": server_args,
                "server_port": server_port,
            }
            client_server_args = {
                "server_type": server_type,
                "port": server_port,
            }
        else:  # model is hosted elsewhere
            client_server_args = {
                "server_type": server_type,
                "host": server_address,
                "model": server_model,
            }
        # TODO: better way to pass arguments?
        cluster_config["required_env_vars"] = cluster_config.get("required_env_vars", []) + [
            f"REWARD_SERVER_ARGS='{json.dumps(client_server_args)}'"
        ]

    with pipeline_utils.get_exp(expname, cluster_config, _reuse_exp) as exp:
        prev_task = _task_dependencies
        for job_id in range(dependent_jobs + 1):
            if job_id == dependent_jobs and convert_last_ckpt_to_hf:
                ckpt_dir = f"{output_dir}/checkpoints"
                actor_dir = f"{ckpt_dir}/global_step_$(<{ckpt_dir}/latest_checkpointed_iteration.txt)/actor"
                convert_cmd = f"python3 -m verl.utils.checkpoint.convert_checkpoint --ckpt_path {actor_dir}"
                hf_input = f"{actor_dir}/huggingface"
                cp_last_ckpt_cmd = f'cp -r "{hf_input}" "{final_ckpt_path}"/'

                train_cmd = f"{train_cmd} && {convert_cmd} && {cp_last_ckpt_cmd}"
            prev_task = pipeline_utils.add_task(
                exp,
                cmd=train_cmd,
                task_name=f"{expname}-ppo-{job_id}",
                log_dir=f"{log_dir}/training-logs",
                container=cluster_config["containers"]["verl"],
                num_gpus=num_gpus,
                num_nodes=num_nodes,
                num_tasks=1,  # torchrun will launch all processes
                cluster_config=cluster_config,
                server_config=server_config,
                partition=partition,
                run_after=run_after,
                reuse_code=reuse_code,
                reuse_code_exp=reuse_code_exp,
                task_dependencies=[prev_task] if prev_task is not None else None,
                sbatch_kwargs=parse_kwargs(sbatch_kwargs, exclusive=exclusive, qos=qos, time_min=time_min),
                heterogeneous=True if server_config is not None else False,
                with_sandbox=with_sandbox,
                installation_command=installation_command,
                skip_hf_home_check=skip_hf_home_check,
            )
        # explicitly setting sequential to False since we set dependencies directly
        pipeline_utils.run_exp(exp, cluster_config, sequential=False, dry_run=dry_run)

    if _reuse_exp:
        return [prev_task]
    return exp


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
