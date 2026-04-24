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

import argparse
import os
import subprocess
from shlex import join


def _sglang_launch_prefix():
    """Return a command prefix that gets the right Python + sglang module.

    SGLang's DeepSeek-V4-hopper image does not expose sglang as an installed
    package by default; instead the source tree may sit under /workspace/sglang
    or /sgl-workspace/sglang. In the Hopper image, system python has the runtime
    deps, while the companion /sgl-workspace/ns-venv lacks some deps, so prefer
    system python with PYTHONPATH pointed at the source tree.
    """
    sglang_src = next(
        (path for path in ["/workspace/sglang/python", "/sgl-workspace/sglang/python"] if os.path.isdir(path)),
        None,
    )
    if sglang_src:
        return f"PYTHONPATH={sglang_src}:$PYTHONPATH python3"
    return "python3"


def main():
    parser = argparse.ArgumentParser(description="Serve SGlang model")
    parser.add_argument("--model", help="Path to the model or a model name to pull from HF")
    parser.add_argument("--num_gpus", type=int, required=True)
    parser.add_argument("--num_nodes", type=int, required=False, default=1)
    parser.add_argument("--node_rank", type=int, required=False)
    parser.add_argument("--dist_init_addr", type=str, required=False)
    parser.add_argument("--tensor_parallel_size", "--tensor-parallel-size", "--tp-size", type=int, required=False)
    parser.add_argument("--data_parallel_size", "--data-parallel-size", "--dp-size", type=int, required=False)
    parser.add_argument("--port", type=int, default=20000, help="Server port")
    args, unknown = parser.parse_known_args()

    if args.num_nodes > 1:
        if args.node_rank is None:
            raise ValueError("node_rank must be specified for multi-node setup")
        if args.dist_init_addr is None:
            raise ValueError("dist_init_addr must be specified for multi-node setup")

    extra_arguments = join(unknown)

    print(f"Deploying model {args.model}")
    print("Starting OpenAI Server")
    tensor_parallel_size = args.tensor_parallel_size or args.num_gpus * args.num_nodes

    multinode_paramaters = (
        f"    --nnodes={args.num_nodes} "
        f"    --node-rank={args.node_rank} "
        f'    --dist-init-addr="{args.dist_init_addr}:20000" '
        if args.num_nodes > 1
        else ""
    )

    cmd = (
        f"{_sglang_launch_prefix()} -m sglang.launch_server "
        f'    --model="{args.model}" '
        f'    --served-model-name="{args.model}"'
        f"    --trust-remote-code "
        f'    --host="0.0.0.0" '
        f"    --port={args.port} "
        f"    --tensor-parallel-size={tensor_parallel_size} "
        f"    {f'--data-parallel-size={args.data_parallel_size}' if args.data_parallel_size else ''} "
        f"    {multinode_paramaters} "
        f"    {extra_arguments} "
    )

    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
