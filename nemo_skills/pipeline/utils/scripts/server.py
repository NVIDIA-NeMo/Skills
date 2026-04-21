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

"""Server and sandbox script classes for NeMo-Skills pipeline."""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from nemo_skills.pipeline.utils.commands import sandbox_command
from nemo_skills.pipeline.utils.scripts.base import BaseJobScript
from nemo_skills.pipeline.utils.server import get_free_port, get_server_command
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


@dataclass(kw_only=True)
class ServerScript(BaseJobScript):
    """Script for model inference servers (vLLM, TRT-LLM, SGLang, etc.).

    This script wraps server command builders and provides:
    - Automatic port allocation if not specified
    - Type-safe server configuration
    - Cross-component address sharing (get_address())
    - Resource requirement tracking (num_gpus, num_nodes, num_tasks)

    Attributes:
        server_type: Type of server (vllm, trtllm, sglang, megatron, openai, etc.)
        model_path: Path to model weights or model name for API services
        cluster_config: Cluster configuration dictionary
        num_gpus: Number of GPUs required (default: 8)
        num_nodes: Number of nodes required (default: 1)
        server_args: Additional server-specific arguments
        server_entrypoint: Custom server entrypoint script (optional)
        port: Server port (allocated automatically if None)
        allocate_port: Whether to allocate port automatically (default: True)
        num_tasks: Number of MPI tasks (computed in __post_init__)
        log_prefix: Prefix for log files (default: "server")

    Example:
        server = ServerScript(
            server_type="vllm",
            model_path="/models/llama-3-8b",
            cluster_config=cluster_config,
            num_gpus=8,
        )
        print(f"Server will run on port {server.port}")
    """

    server_type: str
    model_path: str
    cluster_config: Dict
    num_gpus: int = 8
    num_nodes: int = 1
    server_args: str = ""
    server_entrypoint: Optional[str] = None
    port: Optional[int] = None
    allocate_port: bool = True

    # Server spans all group nodes (e.g., for distributed inference)
    span_group_nodes: bool = True

    # Computed fields (set in __post_init__)
    num_tasks: int = field(init=False, repr=False)
    log_prefix: str = field(default="server", init=False)

    def __post_init__(self):
        if self.port is None and self.allocate_port:
            self.port = get_free_port(strategy="random")
            LOG.debug(f"Allocated port {self.port} for {self.server_type} server")
        if self.port is None:
            raise ValueError("ServerScript requires `port` when allocate_port=False")

        cmd, self.num_tasks = get_server_command(
            server_type=self.server_type,
            num_gpus=self.num_gpus,
            num_nodes=self.num_nodes,
            model_path=self.model_path,
            cluster_config=self.cluster_config,
            server_port=self.port,
            server_args=self.server_args,
            server_entrypoint=self.server_entrypoint,
        )

        self.set_inline(cmd)
        super().__post_init__()

    def get_address(self) -> str:
        """Get server address for client connections (hostname:port)."""
        return f"{self.hostname_ref()}:{self.port}"


@dataclass(kw_only=True)
class SandboxScript(BaseJobScript):
    """Script for code execution sandbox container.

    Attributes:
        cluster_config: Cluster configuration dictionary
        port: Sandbox port (allocated automatically if None)
        keep_mounts: Whether to keep filesystem mounts (default: False, risky if True).
        allocate_port: Whether to allocate port automatically (default: True)
        log_prefix: Prefix for log files (default: "sandbox")
    """

    cluster_config: Dict
    port: Optional[int] = None
    keep_mounts: bool = False
    allocate_port: bool = True
    env_overrides: Optional[List[str]] = None

    # Sandbox spans all group nodes (e.g., for multi-node generate jobs)
    span_group_nodes: bool = True

    log_prefix: str = field(default="sandbox", init=False)

    def __post_init__(self):
        if self.port is None and self.allocate_port:
            self.port = get_free_port(strategy="random")
            LOG.debug(f"Allocated port {self.port} for sandbox")
        if self.port is None:
            raise ValueError("SandboxScript requires `port` when allocate_port=False")

        cmd, metadata = sandbox_command(
            cluster_config=self.cluster_config,
            port=self.port,
        )

        def build_cmd() -> Tuple[str, Dict]:
            env = dict(metadata.get("environment", {}))
            if self.env_overrides:
                for override in self.env_overrides:
                    key, value = override.split("=", 1)
                    env[key] = value
            return cmd, {"environment": env}

        self.set_inline(build_cmd)
        super().__post_init__()
