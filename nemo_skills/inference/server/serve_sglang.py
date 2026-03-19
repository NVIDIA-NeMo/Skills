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
import subprocess
from shlex import join


def main():
    parser = argparse.ArgumentParser(description="Serve SGlang model")
    parser.add_argument("--model", help="Path to the model or a model name to pull from HF")
    parser.add_argument("--num_gpus", type=int, required=True)
    parser.add_argument("--num_nodes", type=int, required=False, default=1)
    parser.add_argument("--node_rank", type=int, required=False)
    parser.add_argument("--dist_init_addr", type=str, required=False)
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

    multinode_paramaters = (
        f"    --nnodes={args.num_nodes} "
        f"    --node-rank={args.node_rank} "
        f'    --dist-init-addr="{args.dist_init_addr}:20000" '
        if args.num_nodes > 1
        else ""
    )

    cmd = (
        f"python3 -m sglang.launch_server "
        f'    --model="{args.model}" '
        f'    --served-model-name="{args.model}"'
        f"    --trust-remote-code "
        f'    --host="0.0.0.0" '
        f"    --port={args.port} "
        f"    --tensor-parallel-size={args.num_gpus * args.num_nodes} "  # TODO: is this a good default for multinode setup?
        f"    {multinode_paramaters} "
        f"    {extra_arguments} "
    )

    subprocess.run(cmd, shell=True, check=True)


if __name__ == "__main__":
    main()
