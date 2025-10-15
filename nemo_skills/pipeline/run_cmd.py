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

import logging
from typing import List

import typer

from nemo_skills.pipeline import utils as pipeline_utils
from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils import add_task, check_mounts, get_exp, run_exp
from nemo_skills.utils import get_logger_name, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


def get_cmd(command):
    cmd = (
        f"export HYDRA_FULL_ERROR=1 && "
        f"export PYTHONPATH=$PYTHONPATH:/nemo_run/code && "
        f"cd /nemo_run/code && "
        f"{command.strip()} "
    )
    return cmd


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@typer_unpacker
def run_cmd(
    ctx: typer.Context,
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    command: str = typer.Option(
        None, help="Command to run in the container. Can also be specified as extra arguments."
    ),
    container: str = typer.Option("nemo-skills", help="Container to use for the run"),
    expname: str = typer.Option("script", help="Nemo run experiment name"),
    partition: str = typer.Option(
        None, help="Can specify if need interactive jobs or a specific non-default partition"
    ),
    qos: str = typer.Option(None, help="Specify Slurm QoS, e.g. to request interactive nodes"),
    time_min: str = typer.Option(None, help="If specified, will use as a time-min slurm parameter"),
    num_gpus: int | None = typer.Option(None, help="Number of GPUs to use"),
    num_nodes: int = typer.Option(1, help="Number of nodes to use"),
    model: str = typer.Option(None, help="Path to the model to evaluate"),
    server_address: str = typer.Option(None, help="Address of the server hosting the model"),
    server_type: pipeline_utils.SupportedServers | None = typer.Option(None, help="Type of server to use"),
    server_gpus: int = typer.Option(None, help="Number of GPUs to use if hosting the model"),
    server_nodes: int = typer.Option(1, help="Number of nodes to use if hosting the model"),
    server_args: str = typer.Option("", help="Additional arguments for the server"),
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
    with_sandbox: bool = typer.Option(False, help="If True, will start a sandbox container alongside this job"),
    keep_mounts_for_sandbox: bool = typer.Option(
        False,
        help="If True, will keep the mounts for the sandbox container. Note that, it is risky given that sandbox executes LLM commands and could potentially lead to data loss. So, we advise not to use this unless absolutely necessary.",
    ),
    log_dir: str = typer.Option(
        None,
        help="Can specify a custom location for slurm logs. "
        "If not specified, will be inside `ssh_tunnel.job_dir` part of your cluster config.",
    ),
    exclusive: bool | None = typer.Option(None, help="If set will add exclusive flag to the slurm job."),
    get_random_port: bool = typer.Option(False, help="If True, will get a random port for the server"),
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
    """Run a pre-defined module or script in the Nemo-Skills container."""
    setup_logging(disable_hydra_logs=False, use_rich=True)
    extra_arguments = f"{' '.join(ctx.args)}"

    # Assert that either command or extra_arguments is provided, not both
    if command and extra_arguments:
        raise ValueError("Please provide either a command or extra arguments, not both.")
    elif not command and not extra_arguments:
        raise ValueError("Please provide either a command or extra arguments.")

    command = command or extra_arguments  # From here on, `command` will be used as the command to run
    extra_arguments = ""  # Reset extra_arguments to avoid confusion

    # Setup cluster config
    cluster_config = pipeline_utils.get_cluster_config(cluster, config_dir)
    cluster_config = pipeline_utils.resolve_mount_paths(
        cluster_config, mount_paths, create_remote_dir=check_mounted_paths
    )

    # we support running multiple commands in their own containers inside a job
    commands = command
    containers = container
    if not isinstance(commands, list):
        commands = [commands]
    if not isinstance(containers, list):
        containers = [containers]

    commands = [get_cmd(cmd) for cmd in commands]
    containers = [cluster_config["containers"].get(container, container) for container in containers]

    if len(commands) != len(containers):
        raise ValueError(
            "If you provide multiple commands, you must also provide the same number of containers to run them in."
        )

    log_dir = check_mounts(cluster_config, log_dir, check_mounted_paths=check_mounted_paths)

    with get_exp(expname, cluster_config, _reuse_exp) as exp:
        # Setup server config if model is provided
        if model is not None:
            server_config, server_address, extra_arguments = pipeline_utils.configure_client(
                model=model,
                server_type=server_type,
                server_address=server_address,
                server_gpus=server_gpus,
                server_nodes=server_nodes,
                server_args=server_args,
                server_entrypoint=server_entrypoint,
                server_container=server_container,
                extra_arguments=extra_arguments,  # this is empty string by design
                get_random_port=get_random_port,
            )
        else:
            server_config = None

        # Wrap command with generation command if model is provided
        if model is not None and server_config is not None:
            commands = [pipeline_utils.set_python_path_and_wait_for_server(server_address, cmd) for cmd in commands]

        prev_tasks = _task_dependencies
        for _ in range(dependent_jobs + 1):
            new_task = add_task(
                exp,
                cmd=commands,
                task_name=expname,
                log_dir=log_dir,
                container=containers,
                cluster_config=cluster_config,
                partition=partition,
                qos=qos,
                time_min=time_min,
                server_config=server_config,
                with_sandbox=with_sandbox,
                keep_mounts_for_sandbox=keep_mounts_for_sandbox,
                sandbox_port=None if get_random_port else 6000,
                run_after=run_after,
                reuse_code=reuse_code,
                reuse_code_exp=reuse_code_exp,
                task_dependencies=prev_tasks,
                num_gpus=num_gpus,
                num_nodes=num_nodes,
                num_tasks=[1] * len(commands),
                slurm_kwargs={"exclusive": exclusive} if exclusive else None,
                installation_command=installation_command,
                skip_hf_home_check=skip_hf_home_check,
            )
            prev_tasks = [new_task]
        run_exp(exp, cluster_config, dry_run=dry_run)

    if _reuse_exp:
        return prev_tasks
    return exp


if __name__ == "__main__":
    typer.main.get_command_name = lambda name: name
    app()
