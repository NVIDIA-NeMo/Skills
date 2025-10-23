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

import argparse
import os
import subprocess


def main():
    parser = argparse.ArgumentParser(description="Serve Riva NIM entrypoint")
    parser.add_argument("--model", required=True, help="Path or identifier of the model")
    parser.add_argument("--num_gpus", type=int, required=True, help="Number of GPUs to use")
    parser.add_argument(
        "--num_nodes",
        type=int,
        required=False,
        default=1,
        help="Not used. Added for compatibility with generic server",
    )
    parser.add_argument("--port", type=int, required=True, help="Base API port (HTTP/GRPC)")

    # Explicit NIM environment options
    parser.add_argument(
        "--nim-tags-selector",
        "--NIM_TAGS_SELECTOR",
        dest="nim_tags_selector",
        type=str,
        default=None,
        help="Tags selector for NIM (e.g., 'name=parakeet-tdt-0.6b-v2,mode=ofl')",
    )
    parser.add_argument(
        "--nim-disable-model-download",
        "--NIM_DISABLE_MODEL_DOWNLOAD",
        dest="nim_disable_model_download",
        type=str,
        choices=["true", "false"],
        default="true",
        help="Whether to disable model download inside NIM (true/false)",
    )
    parser.add_argument(
        "--nim-export-path",
        "--NIM_EXPORT_PATH",
        dest="nim_export_path",
        type=str,
        default=None,
        help="Export path for NIM artifacts (e.g., /mnt_nim_export)",
    )
    parser.add_argument(
        "--container-id",
        "--CONTAINER_ID",
        dest="container_id",
        type=str,
        default=None,
        help="Container/model identifier to select within NIM image",
    )

    args = parser.parse_args()

    # Compute ports
    http_port = int(args.port)
    grpc_port = http_port + 1
    triton_grpc_port = http_port + 2
    triton_http_port = http_port + 3
    triton_metrics_port = http_port + 4

    # Base environment
    env = os.environ.copy()

    # Map known ports
    env.update(
        {
            "NIM_HTTP_API_PORT": str(http_port),
            "NIM_GRPC_API_PORT": str(grpc_port),
            "NIM_GRPC_TRITON_PORT": str(triton_grpc_port),
            "NIM_HTTP_TRITON_PORT": str(triton_http_port),
            "NIM_TRITON_METRICS_PORT": str(triton_metrics_port),
        }
    )

    # Helpful defaults unless overridden by user-provided options
    env.setdefault("NIM_DISABLE_MODEL_DOWNLOAD", "true")

    # Convert any extra options to env vars
    # Map explicit options to env
    if args.nim_tags_selector is not None:
        env["NIM_TAGS_SELECTOR"] = args.nim_tags_selector
    if args.nim_disable_model_download is not None:
        env["NIM_DISABLE_MODEL_DOWNLOAD"] = args.nim_disable_model_download
    if args.nim_export_path is not None:
        env["NIM_EXPORT_PATH"] = args.nim_export_path
    if args.container_id is not None:
        env["CONTAINER_ID"] = args.container_id

    # Launch the NIM inside the container
    cmd = "/opt/nim/start_server.sh"
    print(
        "Starting Riva NIM with ports: HTTP=%s, GRPC=%s, TRITON(g/h/m)=(%s,%s,%s)"
        % (http_port, grpc_port, triton_grpc_port, triton_http_port, triton_metrics_port)
    )
    sanitized = {k: v for k, v in env.items() if k.startswith(("NIM_", "CONTAINER_ID"))}
    print(f"Environment (sanitized): {sanitized}")
    subprocess.run(cmd, shell=True, check=True, env=env)


if __name__ == "__main__":
    main()
