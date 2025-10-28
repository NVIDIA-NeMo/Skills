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
import typer

from nemo_skills.pipeline.app import app, typer_unpacker
from nemo_skills.pipeline.utils import (
    SupportedServersSelfHosted,
    add_task,
    check_mounts,
    get_cluster_config,
    get_exp,
    get_free_port,
    parse_sbatch_arguments,
    resolve_mount_paths,
    set_python_path_and_wait_for_server,
)
from nemo_skills.utils import setup_logging


def get_gradio_chat_cmd(model, server_type, extra_args):
    cmd = (
        "python -m nemo_skills.inference.chat_interface.launch "
        f"model_config_path={model}/config.json "
        f"server_type={server_type} "
        f" {extra_args} "
    )
    return cmd


@app.command()
@typer_unpacker
def start_server(
    cluster: str = typer.Option(
        None,
        help="One of the configs inside config_dir or NEMO_SKILLS_CONFIG_DIR or ./cluster_configs. "
        "Can also use NEMO_SKILLS_CONFIG instead of specifying as argument.",
    ),
    model: str = typer.Option(..., help="Path to the model"),
    server_type: SupportedServersSelfHosted = typer.Option(..., help="Type of server to use"),
    server_gpus: int = typer.Option(..., help="Number of GPUs to use for hosting the model"),
    server_nodes: int = typer.Option(1, help="Number of nodes to use for hosting the model"),
    server_args: str = typer.Option("", help="Additional arguments for the server"),
    server_entrypoint: str = typer.Option(
        None,
        help="Path to the entrypoint of the server. "
        "If not specified, will use the default entrypoint for the server type.",
    ),
    server_container: str = typer.Option(None, help="Override container image for the hosted server"),
    partition: str = typer.Option(None, help="Cluster partition to use"),
    qos: str = typer.Option(None, help="Specify Slurm QoS, e.g. to request interactive nodes"),
    time_min: str = typer.Option(None, help="If specified, will use as a time-min slurm parameter"),
    mount_paths: str = typer.Option(None, help="Comma separated list of paths to mount on the remote machine"),
    with_sandbox: bool = typer.Option(
        False, help="Starts a sandbox (set this flag if model supports calling Python interpreter)"
    ),
    keep_mounts_for_sandbox: bool = typer.Option(
        False,
        help="If True, will keep the mounts for the sandbox container. Note that, it is risky given that sandbox executes LLM commands and could potentially lead to data loss. So, we advise not to use this unless absolutely necessary.",
    ),
    launch_chat_interface: bool = typer.Option(
        False, help="If True, will launch a gradio app that provides chat with the model"
    ),
    extra_chat_args: str = typer.Option("", help="Extra hydra arguments to be passed to the chat app"),
    config_dir: str = typer.Option(None, help="Can customize where we search for cluster configs"),
    log_dir: str = typer.Option(
        None,
        help="Can specify a custom location for slurm logs. "
        "If not specified, will be inside `ssh_tunnel.job_dir` part of your cluster config.",
    ),
    exclusive: bool = typer.Option(False, help="If set will add exclusive flag to the slurm job."),
    check_mounted_paths: bool = typer.Option(False, help="Check if mounted paths are available on the remote machine"),
    get_random_port: bool = typer.Option(False, help="If True, will get a random port for the server"),
    sbatch_arguments: str = typer.Option(
        "",
        help="Additional sbatch arguments to pass to the job scheduler. Values should be provided as a JSON string or as a `dict` if invoking from code.",
    ),
):
    """Self-host a model server."""
    setup_logging(disable_hydra_logs=False, use_rich=True)

    cluster_config = get_cluster_config(cluster, config_dir)
    cluster_config = resolve_mount_paths(cluster_config, mount_paths)

    try:
        server_type = server_type.value
    except AttributeError:
        pass

    log_dir = check_mounts(cluster_config, log_dir, check_mounted_paths=check_mounted_paths)

    server_config = {
        "model_path": model,
        "server_type": server_type,
        "num_gpus": server_gpus,
        "num_nodes": server_nodes,
        "server_args": server_args,
        "server_entrypoint": server_entrypoint,
        "server_port": get_free_port(strategy="random") if get_random_port else 5000,
    }
    if server_container:
        server_config["container"] = server_container

    with get_exp("server", cluster_config) as exp:
        cmd = ""
        if launch_chat_interface:
            server_address = f"localhost:{server_config['server_port']}"
            cmd = set_python_path_and_wait_for_server(
                server_address, get_gradio_chat_cmd(model, server_type, extra_chat_args)
            )
        add_task(
            exp,
            cmd=cmd,
            task_name="server",
            log_dir=log_dir,
            container=cluster_config["containers"]["nemo-skills"],
            cluster_config=cluster_config,
            partition=partition,
            qos=qos,
            time_min=time_min,
            server_config=server_config,
            with_sandbox=with_sandbox,
            keep_mounts_for_sandbox=keep_mounts_for_sandbox,
            sandbox_port=None if get_random_port else 6000,
            slurm_kwargs=parse_sbatch_arguments(sbatch_arguments, exclusive),
        )
        # we don't want to detach in this case even on slurm, so not using run_exp
        exp.run(detach=False, tail_logs=True)
        # TODO: seems like not being killed? If nemorun doesn't do this, we can catch the signal and kill the server ourselves


if __name__ == "__main__":
    # workaround for https://github.com/fastapi/typer/issues/341
    typer.main.get_command_name = lambda name: name
    app()
