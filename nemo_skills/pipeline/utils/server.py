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
import shlex
from enum import Enum

from nemo_skills.pipeline.utils.mounts import check_if_mounted
from nemo_skills.utils import get_logger_name, get_server_wait_cmd

LOG = logging.getLogger(get_logger_name(__file__))


class SupportedServersSelfHosted(str, Enum):
    trtllm = "trtllm"
    vllm = "vllm"
    vllm_dp_ray = "vllm_dp_ray"
    vllm_multimodal = "vllm_multimodal"
    sglang = "sglang"
    megatron = "megatron"
    generic = "generic"


class SupportedServers(str, Enum):
    trtllm = "trtllm"
    vllm = "vllm"
    vllm_dp_ray = "vllm_dp_ray"
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


def _parse_last_flag(tokens: list[str], *names: str) -> str | None:
    """Return the last matched value for any of the given CLI flag names.

    Accepts both ``--flag value`` and ``--flag=value`` forms. Searches left
    to right and returns the last match, matching the shell's "last wins"
    semantics when a flag is specified more than once.
    """
    last_value: str | None = None
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        matched = False
        for name in names:
            if tok == name:
                if i + 1 < len(tokens):
                    last_value = tokens[i + 1]
                i += 2
                matched = True
                break
            if tok.startswith(name + "="):
                last_value = tok.split("=", 1)[1]
                i += 1
                matched = True
                break
        if not matched:
            i += 1
    return last_value


def _compute_vllm_dp_ray_serving_nodes(server_args: str, num_gpus: int, num_nodes: int) -> int:
    """Compute how many SLURM ranks participate in vLLM serving for vllm_dp_ray.

    Parses ``--tensor-parallel-size``, ``--pipeline-parallel-size``, and
    ``--data-parallel-size`` out of ``server_args`` (shlex-split, last match
    wins) and returns ``dp_size × ceil(tp*pp / num_gpus)``. Raises
    ``ValueError`` on inconsistent topology.
    """
    tokens = shlex.split(server_args or "")
    tp = int(_parse_last_flag(tokens, "--tensor-parallel-size", "--tensor_parallel_size", "-tp") or 1)
    pp = int(_parse_last_flag(tokens, "--pipeline-parallel-size", "--pipeline_parallel_size", "-pp") or 1)
    dp_raw = _parse_last_flag(tokens, "--data-parallel-size", "--data_parallel_size", "-dp")
    if dp_raw is None:
        raise ValueError(
            "server_type=vllm_dp_ray requires --data-parallel-size to be set explicitly "
            "in server_args. For single-replica serving use server_type=vllm instead."
        )
    dp = int(dp_raw)
    world_size = tp * pp
    if world_size > num_gpus:
        if world_size % num_gpus != 0:
            raise ValueError(
                f"vllm_dp_ray: world_size={world_size} (tp={tp} × pp={pp}) must be an integer "
                f"multiple of num_gpus={num_gpus} for multi-node DP replicas."
            )
        replica_nodes = world_size // num_gpus
    else:
        replica_nodes = 1
    serving_nodes = dp * replica_nodes
    if serving_nodes > num_nodes:
        raise ValueError(
            f"vllm_dp_ray: dp={dp} × replica_nodes={replica_nodes} = {serving_nodes} serving nodes "
            f"exceeds num_nodes={num_nodes}. Allocate more nodes, reduce dp, or reduce tp*pp."
        )
    return serving_nodes


def get_ray_server_cmd(start_cmd, serving_nodes: int | None = None, num_gpus: int | None = None):
    """Build the Ray-cluster startup script for multi-node serving.

    If ``serving_nodes`` is provided and < total num_nodes, workers with
    ``SLURM_PROCID >= serving_nodes`` join Ray with ``--num-gpus=0`` and a
    custom ``extra_gpu`` resource advertising ``num_gpus`` slots. This hides
    their GPUs from vLLM's GPU discovery (so DP placement/coordination
    ignores them) while still exposing a resource that opt-in Ray tasks
    (e.g. neural-eval metrics like xCOMET) can schedule against via
    ``@ray.remote(num_gpus=0, resources={"extra_gpu": 1})``. Such tasks
    must also set ``RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1`` on
    the node so physical GPUs remain visible to the task process.

    ``serving_nodes`` counts all ranks that participate in vLLM serving,
    i.e. ``data_parallel_size × ceil(tp*pp / num_gpus)`` — not just
    ``data_parallel_size``, so multi-node DP replicas (tp*pp > num_gpus)
    are handled correctly.
    """
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

    if serving_nodes is not None:
        if num_gpus is None:
            raise ValueError("get_ray_server_cmd: num_gpus must be set when serving_nodes is set")
        # Worker branch: split on whether this rank participates in vLLM serving.
        worker_branch = (
            'if [ "${SLURM_PROCID:-0}" -lt ' + str(serving_nodes) + " ]; then "
            "    echo 'Starting serving worker node' && "
            "    ray start "
            "        --block "
            "        --address=$SLURM_MASTER_NODE:6379 "
            f"       {ports} ;"
            "else "
            "    echo 'Starting extra worker node (GPUs hidden from vLLM)' && "
            "    export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 && "
            "    ray start "
            "        --block "
            "        --address=$SLURM_MASTER_NODE:6379 "
            "        --num-gpus=0 "
            f"       --resources='{{\"extra_gpu\": {num_gpus}}}' "
            f"       {ports} ;"
            "fi"
        )
    else:
        worker_branch = (
            "    echo 'Starting worker node' && "
            '    echo "Connecting to head node at $SLURM_MASTER_NODE" && '
            "    ray start "
            "        --block "
            "        --address=$SLURM_MASTER_NODE:6379 "
            f"       {ports} ;"
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
        "    export RAY_raylet_start_wait_time_s=120 && "
        f"   {worker_branch} "
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
    if server_type not in ["vllm", "vllm_dp_ray", "vllm_multimodal", "sglang", "trtllm", "generic"]:
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
    elif server_type == "vllm_dp_ray":
        # Same interface as `vllm`, but starts vLLM in-process on the Ray head
        # so NeMo Gym's DP-on-Ray placement-group monkey-patch takes effect.
        # Required when total Ray nodes > serving_nodes (e.g. an extra
        # node reserved for a Ray-scheduled neural-eval task), and also
        # supports multi-node DP replicas (tp*pp > num_gpus per node).
        server_entrypoint = server_entrypoint or "-m nemo_skills.inference.server.serve_vllm_dp_ray"
        start_vllm_cmd = (
            f"python3 {server_entrypoint} "
            f"    --model {model_path} "
            f"    --num_gpus {num_gpus} "
            f"    --num_nodes {num_nodes} "
            f"    --port {server_port} "
            f"    {server_args} "
        )
        serving_nodes = _compute_vllm_dp_ray_serving_nodes(server_args, num_gpus, num_nodes)

        # Ray must be started externally for this path: the wrapper calls
        # ray.init(address="auto") and relies on an already-running cluster
        # so it can pre-reserve DP rank 0's PG before vLLM boots.
        server_start_cmd = get_ray_server_cmd(
            start_vllm_cmd,
            serving_nodes=serving_nodes,
            num_gpus=num_gpus,
        )
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
        if num_nodes > 1 and server_entrypoint in ("trtllm", "trtllm-serve"):
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
