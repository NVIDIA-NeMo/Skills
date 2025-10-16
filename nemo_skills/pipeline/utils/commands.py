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

"""
Minimal command builders for declarative pipeline interface.

These are thin wrappers around existing command construction utilities
in server.py and other modules, designed to work with Command objects.
"""

from typing import Dict, Optional, Tuple

from nemo_skills.pipeline.utils.exp import get_sandbox_command
from nemo_skills.pipeline.utils.server import get_free_port, get_server_command


def vllm_server_command(
    cluster_config: Dict,
    model: str,
    port: Optional[int] = None,
    server_type: str = "vllm",
    gpus: int = 8,
    nodes: int = 1,
    args: str = "",
    entrypoint: Optional[str] = None,
    **kwargs,
) -> Tuple[str, Dict]:
    """Build vLLM server command.

    Args:
        cluster_config: Cluster configuration dictionary
        model: Model path or name
        port: Port to use (if None, will use get_free_port)
        server_type: Type of server (vllm, sglang, trtllm, megatron)
        gpus: Number of GPUs
        nodes: Number of nodes
        args: Additional server arguments
        entrypoint: Custom entrypoint script

    Returns:
        Tuple of (command_string, metadata_dict)
    """
    if port is None:
        port = get_free_port(strategy="random")

    cmd, num_tasks = get_server_command(
        server_type=server_type,
        num_gpus=gpus,
        num_nodes=nodes,
        model_path=model,
        cluster_config=cluster_config,
        server_port=port,
        server_args=args,
        server_entrypoint=entrypoint,
    )

    metadata = {
        "port": port,
        "log_prefix": "server",
        "num_tasks": num_tasks,
    }

    return cmd, metadata


def sandbox_command(cluster_config: Dict, port: int, **kwargs) -> Tuple[str, Dict]:
    """Build sandbox command.

    Args:
        cluster_config: Cluster configuration dictionary
        port: Port to use for sandbox (required - must be coordinated with client)

    Returns:
        Tuple of (command_string, metadata_dict)

    Note:
        The port must be passed to both sandbox_command() and set as NEMO_SKILLS_SANDBOX_PORT
        for the client command so they can communicate.
    """
    cmd = get_sandbox_command(cluster_config)

    # Build PYTHONPATH from cluster config
    pythonpath_env = {}
    for env_var in cluster_config.get("env_vars", []):
        if "PYTHONPATH" in env_var:
            pythonpath = env_var[11:] if env_var.startswith("PYTHONPATH=") else env_var
            pythonpath_env["PYTHONPATH"] = pythonpath + ":/app"
            break

    metadata = {
        "port": port,
        "log_prefix": "sandbox",
        "environment": {
            "LISTEN_PORT": str(port),
            "NGINX_PORT": str(port),
            **pythonpath_env,
        },
    }

    return cmd, metadata


def wrap_command(command: str, working_dir: str = "/nemo_run/code", env_vars: Optional[Dict[str, str]] = None) -> str:
    """Wrap command with working directory and environment variable setup.

    Args:
        command: The command to wrap
        working_dir: Working directory to cd into
        env_vars: Environment variables to export

    Returns:
        Wrapped command string
    """
    parts = []

    # Export environment variables
    if env_vars:
        for key, value in env_vars.items():
            parts.append(f"export {key}={value}")

    # Change to working directory
    if working_dir:
        parts.append(f"cd {working_dir}")

    # Add the actual command
    parts.append(command)

    return " && ".join(parts)
