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
from enum import Enum

from nemo_skills.pipeline.utils.mounts import check_if_mounted
from nemo_skills.utils import get_logger_name, get_server_wait_cmd

LOG = logging.getLogger(get_logger_name(__file__))


class SupportedServersSelfHosted(str, Enum):
    trtllm = "trtllm"
    vllm = "vllm"
    vllm_multimodal = "vllm_multimodal"
    sglang = "sglang"
    megatron = "megatron"
    generic = "generic"


class SupportedServers(str, Enum):
    trtllm = "trtllm"
    vllm = "vllm"
    vllm_multimodal = "vllm_multimodal"
    sglang = "sglang"
    megatron = "megatron"
    openai = "openai"
    azureopenai = "azureopenai"
    gemini = "gemini"
    generic = "generic"


def get_free_port(exclude: list[int] | None = None, strategy: int | str = 5000) -> int:
    """Will return a free port on the host."""
    exclude = exclude or []
    if isinstance(strategy, int):
        port = strategy
        while port in exclude:
            port += 1
        return port
    elif strategy == "random":
        import random

        port = random.randint(1024, 65535)
        while port in exclude:
            port = random.randint(1024, 65535)
        return port
    else:
        raise ValueError(f"Strategy {strategy} not supported.")


def should_get_random_port(server_gpus, exclusive):
    return server_gpus != 8 and not exclusive


def wrap_python_path(cmd):
    return "export PYTHONPATH=$PYTHONPATH:/nemo_run/code && cd /nemo_run/code && " + cmd


def set_python_path_and_wait_for_server(server_address, generation_commands):
    if server_address is not None:
        cmd = get_server_wait_cmd(server_address) + " && "
    else:
        cmd = ""
    # will run in a single task always (no need to check mpi env vars)
    cmd += f"{generation_commands}"
    return wrap_python_path(cmd)


def get_ray_server_cmd(start_cmd):
    ports = (
        "--node-manager-port=12345 "
        "--object-manager-port=12346 "
        "--dashboard-port=8265 "
        "--dashboard-agent-grpc-port=12347 "
        "--runtime-env-agent-port=12349 "
        "--metrics-export-port=12350 "
        "--min-worker-port=14349 "
        "--max-worker-port=18349 "
    )

    ray_start_cmd = (
        'if [ "${SLURM_PROCID:-0}" = 0 ]; then '
        "    echo 'Starting head node' && "
        "    export RAY_raylet_start_wait_time_s=120 && "
        "    ray start "
        "        --head "
        "        --port=6379 "
        f"       {ports} && "
        f"   {start_cmd} ; "
        "else "
        "    echo 'Starting worker node' && "
        "    export RAY_raylet_start_wait_time_s=120 && "
        '    echo "Connecting to head node at $SLURM_MASTER_NODE" && '
        "    ray start "
        "        --block "
        "        --address=$SLURM_MASTER_NODE:6379 "
        f"       {ports} ;"
        "fi"
    )
    return ray_start_cmd


def get_server_command(
    server_type: str,
    num_gpus: int,
    num_nodes: int,
    model_path: str,
    cluster_config: dict,
    server_port: int,
    server_args: str = "",
    server_entrypoint: str | None = None,
):
    num_tasks = num_gpus

    # check if the model path is mounted if not vllm, sglang, or trtllm;
    # vllm, sglang, trtllm can also pass model name as "model_path" so we need special processing
    if server_type not in ["vllm", "vllm_multimodal", "sglang", "trtllm", "generic"]:
        check_if_mounted(cluster_config, model_path)

    # the model path will be mounted, so generally it will start with /
    elif model_path.startswith("/"):
        check_if_mounted(cluster_config, model_path)

    if server_type == "megatron":
        if cluster_config["executor"] != "slurm":
            num_tasks = 1
            prefix = f"torchrun --nproc_per_node {num_gpus}"
        else:
            prefix = "python "
        server_entrypoint = server_entrypoint or "tools/run_text_generation_server.py"
        # Similar to conversion, we don't hold scripts for megatron on our side
        # and expect it to be in /opt/Megatron-LM in the container
        import os

        megatron_path = os.getenv("MEGATRON_PATH", "/opt/Megatron-LM")
        server_start_cmd = (
            f"export PYTHONPATH=$PYTHONPATH:{megatron_path} && "
            f"export CUDA_DEVICE_MAX_CONNECTIONS=1 && "
            f"cd {megatron_path} && "
            f"{prefix} {server_entrypoint} "
            f"    --load {model_path} "
            f"    --tensor-model-parallel-size {num_gpus} "
            f"    --pipeline-model-parallel-size {num_nodes} "
            f"    --use-checkpoint-args "
            f"    --max-tokens-to-oom 12000000 "
            f"    --port {server_port} "
            f"    --micro-batch-size 1 "  # that's a training argument, ignored here, but required to specify..
            f"    {server_args} "
        )
    elif server_type in ["vllm", "vllm_multimodal"]:
        # vllm_multimodal uses the same vLLM server; multimodal handling is on the client side
        server_entrypoint = server_entrypoint or "-m nemo_skills.inference.server.serve_vllm"
        start_vllm_cmd = (
            f"python3 {server_entrypoint} "
            f"    --model {model_path} "
            f"    --num_gpus {num_gpus} "
            f"    --num_nodes {num_nodes} "
            f"    --port {server_port} "
            f"    {server_args} "
        )
        if num_nodes > 1:
            server_start_cmd = get_ray_server_cmd(start_vllm_cmd)
        else:
            server_start_cmd = start_vllm_cmd
        num_tasks = 1
    elif server_type == "sglang":
        if num_nodes > 1:
            multinode_args = " --dist_init_addr $SLURM_MASTER_NODE --node_rank $SLURM_PROCID "
        else:
            multinode_args = ""
        server_entrypoint = server_entrypoint or "-m nemo_skills.inference.server.serve_sglang"
        server_start_cmd = (
            f"python3 {server_entrypoint} "
            f"    --model {model_path} "
            f"    --num_gpus {num_gpus} "
            f"    --num_nodes {num_nodes} "
            f"    --port {server_port} "
            f"    {multinode_args} "
            f"    {server_args} "
        )
        num_tasks = 1
    elif server_type == "trtllm":
        server_entrypoint = server_entrypoint or "trtllm-serve"
        if num_nodes > 1 and server_entrypoint == "trtllm":
            server_entrypoint = f"trtllm-llmapi-launch {server_entrypoint}"
        else:
            server_entrypoint = f"mpirun -n 1 --oversubscribe --allow-run-as-root {server_entrypoint}"
        server_start_cmd = (
            f"{server_entrypoint} "
            f"    {model_path} "
            f"    --port {server_port} "
            f"    --tp_size {num_gpus * num_nodes} "
            f"    {server_args} "
        )
        if num_nodes == 1:
            num_tasks = 1
        else:
            num_tasks = num_gpus
    elif server_type == "generic":
        if not server_entrypoint:
            raise ValueError("For 'generic' server type, 'server_entrypoint' must be specified.")
        server_start_cmd = (
            f"{server_entrypoint} "
            f"    --model {model_path} "
            f"    --num_gpus {num_gpus} "
            f"    --num_nodes {num_nodes} "
            f"    --port {server_port} "
            f"    {server_args} "
        )
        num_tasks = 1
    else:
        raise ValueError(f"Server type '{server_type}' not supported for model inference.")

    server_cmd = (
        f"nvidia-smi && cd /nemo_run/code && export PYTHONPATH=$PYTHONPATH:/nemo_run/code && {server_start_cmd} "
    )
    return server_cmd, num_tasks
