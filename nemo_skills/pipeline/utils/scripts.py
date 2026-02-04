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
Script classes for NeMo-Skills pipeline components.

These classes wrap NeMo-Run's run.Script interface to provide typed, reusable
job components (servers, clients, sandboxes) with explicit fields and
cross-component reference support for heterogeneous jobs.

Example:
    # Create a server script with automatic port allocation
    server = ServerScript(
        server_type="vllm",
        model_path="/models/llama-8b",
        cluster_config=cluster_config,
        num_gpus=8,
    )

    # Create a client that references the server
    client = GenerationClientScript(
        output_dir="/results",
        input_file="/data/input.jsonl",
        server=server,  # Cross-component reference
    )

    # Use in Command objects
    Command(script=server, container="vllm", ...)
    Command(script=client, container="nemo-skills", ...)
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

import nemo_run as run

from nemo_skills.pipeline.utils.commands import sandbox_command
from nemo_skills.pipeline.utils.exp import install_packages_wrap
from nemo_skills.pipeline.utils.generation import get_generation_cmd
from nemo_skills.pipeline.utils.server import get_free_port, get_server_command
from nemo_skills.utils import get_logger_name

if TYPE_CHECKING:
    # Avoid circular imports for type hints
    pass

LOG = logging.getLogger(get_logger_name(__file__))


@dataclass
class BaseJobScript(run.Script):
    """Base class for job component scripts with heterogeneous job support.

    This class provides:
    - het_group_index tracking for cross-component references in heterogeneous SLURM jobs
    - hostname_ref() method for getting hostnames in het jobs
    - Common pattern for Script initialization

    Attributes:
        het_group_index: Index in heterogeneous job group (set by Pipeline at runtime)
        span_group_nodes: Whether to span all nodes from the group's HardwareConfig.
            When False (default), the script runs on 1 node regardless of group config.
            When True, the script spans all nodes specified in the group's num_nodes.
            This is important for multi-node setups with --overlap where the server
            needs multiple nodes but client/sandbox should run on the master node only.
    """

    het_group_index: Optional[int] = field(default=None, init=False, repr=False)
    span_group_nodes: bool = False  # Default: run on 1 node
    installation_command: Optional[str] = None
    entrypoint: str = field(default="bash", init=False)

    def __post_init__(self):
        """Wrap inline command with installation_command if provided."""
        if not self.installation_command:
            return

        if callable(self.inline):
            original_inline = self.inline

            def wrapped_inline():
                result = original_inline()
                if isinstance(result, tuple):
                    command, metadata = result
                    return install_packages_wrap(command, self.installation_command), metadata
                return install_packages_wrap(result, self.installation_command)

            self.set_inline(wrapped_inline)
        elif isinstance(self.inline, str):
            self.set_inline(install_packages_wrap(self.inline, self.installation_command))

    def set_inline(self, command: Union[str, Callable, run.Script]) -> None:
        """Set the inline command safely on frozen dataclass."""
        object.__setattr__(self, "inline", command)

    def hostname_ref(self) -> str:
        """Get hostname reference for hetjob cross-component communication.

        Returns a shell variable reference that resolves to the master node hostname
        for this het group. Uses environment variables automatically exported by nemo-run:
            SLURM_MASTER_NODE_HET_GROUP_0, SLURM_MASTER_NODE_HET_GROUP_1, etc.

        These are set via:
            export SLURM_MASTER_NODE_HET_GROUP_N=$(scontrol show hostnames $SLURM_JOB_NODELIST_HET_GROUP_N | head -n1)
        """
        if self.het_group_index is None:
            return "127.0.0.1"  # Local fallback for non-heterogeneous jobs

        # Use the environment variable exported by nemo-run
        return f"${{SLURM_MASTER_NODE_HET_GROUP_{self.het_group_index}:-localhost}}"


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
        # Basic usage
        server = ServerScript(
            server_type="vllm",
            model_path="/models/llama-3-8b",
            cluster_config=cluster_config,
            num_gpus=8,
        )

        # Access allocated port
        print(f"Server will run on port {server.port}")

        # Get full address for client connection
        address = server.get_address()  # Returns "hostname:port"
    """

    server_type: str
    model_path: str
    cluster_config: Dict
    num_gpus: int = 8
    num_nodes: int = 1
    server_args: str = ""
    server_entrypoint: Optional[str] = None  # Custom server entrypoint script
    port: Optional[int] = None
    allocate_port: bool = True

    # Server spans all group nodes (e.g., for distributed inference)
    span_group_nodes: bool = True

    # Computed fields (set in __post_init__)
    num_tasks: int = field(init=False, repr=False)
    log_prefix: str = field(default="server", init=False)

    def __post_init__(self):
        """Initialize server script.

        - Allocates port if not provided
        - Builds server command using get_server_command()
        - Sets self.inline to the command string
        - Computes num_tasks from server command builder
        """
        # Allocate port if not provided
        if self.port is None and self.allocate_port:
            self.port = get_free_port(strategy="random")
            LOG.debug(f"Allocated port {self.port} for {self.server_type} server")

        # Build server command
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
        """Get server address for client connections.

        Returns hostname:port string that clients can use to connect.
        In heterogeneous jobs, hostname_ref() returns a bash expression
        that resolves at runtime.

        Returns:
            Server address in format "hostname:port"

        Example:
            # Use in client command
            client_cmd = f"python client.py --server-url http://{server.get_address()}"
        """
        return f"{self.hostname_ref()}:{self.port}"


@dataclass(kw_only=True)
class SandboxScript(BaseJobScript):
    """Script for code execution sandbox container.

    The sandbox provides a secure environment for executing LLM-generated code.
    This script wraps sandbox command builders and provides:
    - Automatic port allocation
    - Mount configuration (can optionally keep mounts, though risky)
    - Type-safe sandbox configuration

    Attributes:
        cluster_config: Cluster configuration dictionary
        port: Sandbox port (allocated automatically if None)
        keep_mounts: Whether to keep filesystem mounts (default: False, risky if True).
                     Note: This is stored for documentation but actually handled at
                     the executor level, not in the sandbox command itself.
        allocate_port: Whether to allocate port automatically (default: True)
        log_prefix: Prefix for log files (default: "sandbox")

    Example:
        sandbox = SandboxScript(
            cluster_config=cluster_config,
            keep_mounts=False,  # Safer: sandbox has no access to mounted paths
        )

        # Client can reference sandbox port
        client = GenerationClientScript(..., sandbox=sandbox)
    """

    cluster_config: Dict
    port: Optional[int] = None
    keep_mounts: bool = False
    allocate_port: bool = True
    env_overrides: Optional[List[str]] = None  # Extra env vars in KEY=VALUE form
    log_prefix: str = field(default="sandbox", init=False)

    def __post_init__(self):
        """Initialize sandbox script.

        - Allocates port if not provided
        - Builds sandbox command using sandbox_command()
        - Sets self.inline to a callable that returns command and environment vars
        """
        # Allocate port if not provided
        if self.port is None and self.allocate_port:
            self.port = get_free_port(strategy="random")
            LOG.debug(f"Allocated port {self.port} for sandbox")

        # Build sandbox command and metadata (including environment vars)
        # Note: keep_mounts is handled at the executor level, not in the command itself
        cmd, metadata = sandbox_command(
            cluster_config=self.cluster_config,
            port=self.port,
        )

        # Use a callable to return both command and environment variables
        # This ensures the sandbox's LISTEN_PORT and NGINX_PORT are properly set
        def build_cmd() -> Tuple[str, Dict]:
            env = dict(metadata.get("environment", {}))
            # Apply user-specified environment overrides
            if self.env_overrides:
                for override in self.env_overrides:
                    key, value = override.split("=", 1)
                    env[key] = value
            return cmd, {"environment": env}

        self.set_inline(build_cmd)
        super().__post_init__()


@dataclass(kw_only=True)
class GenerationClientScript(BaseJobScript):
    """Script for LLM generation/inference client.

    This script wraps generation command builders and provides:
    - Cross-component references to multiple servers and sandbox
    - Lazy command building for runtime hostname resolution
    - Type-safe generation configuration
    - Environment variable handling for sandbox/server communication

    Attributes:
        output_dir: Directory for output files
        input_file: Input JSONL file (mutually exclusive with input_dir)
        input_dir: Input directory (mutually exclusive with input_file)
        extra_arguments: Additional arguments for generation script
        random_seed: Random seed for sampling (optional)
        chunk_id: Chunk ID for parallel processing (optional)
        num_chunks: Total number of chunks (required if chunk_id set)
        preprocess_cmd: Command to run before generation (optional)
        postprocess_cmd: Command to run after generation (optional)
        wandb_parameters: WandB logging configuration (optional)
        with_sandbox: Whether sandbox is enabled
        script: Module or file path for generation script (default: nemo_skills.inference.generate)
        servers: List of ServerScript references (None for pre-hosted servers)
        server_addresses_prehosted: Addresses for pre-hosted servers (parallel to servers list)
        model_names: Model names for multi-model generation (optional)
        server_types: Server types for multi-model generation (optional)
        sandbox: Reference to SandboxScript for cross-component communication (optional)
        log_prefix: Prefix for log files (default: "main")

    Examples:
        # Single server
        client = GenerationClientScript(
            output_dir="/results",
            input_file="/data/input.jsonl",
            servers=[server_script],
            model_names=["llama-8b"],
            server_types=["vllm"],
        )

        # Multi-model with self-hosted and pre-hosted servers
        client = GenerationClientScript(
            output_dir="/results",
            input_file="/data/input.jsonl",
            servers=[server1, server2, None],  # None = pre-hosted
            server_addresses_prehosted=["", "", "https://api.openai.com"],
            model_names=["llama-8b", "llama-70b", "gpt-4"],
            server_types=["vllm", "vllm", "openai"],
            sandbox=sandbox_script,
            with_sandbox=True,
        )
    """

    output_dir: str
    input_file: Optional[str] = None
    input_dir: Optional[str] = None
    extra_arguments: str = ""
    random_seed: Optional[int] = None
    chunk_id: Optional[int] = None
    num_chunks: Optional[int] = None
    preprocess_cmd: Optional[str] = None
    postprocess_cmd: Optional[str] = None
    wandb_parameters: Optional[Dict] = None
    with_sandbox: bool = False
    script: str = "nemo_skills.inference.generate"
    requirements: Optional[list[str]] = None

    # Cross-component references for single/multi-model
    servers: Optional[List[Optional["ServerScript"]]] = None
    server_addresses_prehosted: Optional[List[str]] = None
    model_names: Optional[List[str]] = None
    server_types: Optional[List[str]] = None
    sandbox: Optional["SandboxScript"] = None

    log_prefix: str = field(default="main", init=False)

    def __post_init__(self):
        """Initialize generation client script with lazy command building.

        Builds command lazily via a callable that is evaluated when het_group_index
        is assigned, allowing hostname_ref() to resolve correctly for heterogeneous jobs.

        This works for both cases:
        - With cross-refs: Resolves server hostnames and sandbox ports at runtime
        - Without cross-refs: Just builds the command string (no runtime resolution needed)
        """

        def build_cmd() -> Tuple[str, Dict]:
            """Build command at runtime when cross-refs are resolved."""
            env_vars = {}

            # Add sandbox port to environment if sandbox is referenced
            if self.sandbox:
                env_vars["NEMO_SKILLS_SANDBOX_PORT"] = str(self.sandbox.port)

            # Build server addresses if servers are provided
            server_addresses = None
            if self.servers is not None:
                server_addresses = []
                for server_idx, server_script in enumerate(self.servers):
                    if server_script is not None:
                        # Self-hosted: construct address from hostname and port refs
                        addr = f"{server_script.hostname_ref()}:{server_script.port}"
                    else:
                        # Pre-hosted: use the address from server_addresses_prehosted
                        addr = self.server_addresses_prehosted[server_idx]
                    server_addresses.append(addr)

            # Build generation command
            cmd = get_generation_cmd(
                output_dir=self.output_dir,
                input_file=self.input_file,
                input_dir=self.input_dir,
                extra_arguments=self.extra_arguments,
                random_seed=self.random_seed,
                chunk_id=self.chunk_id,
                num_chunks=self.num_chunks,
                preprocess_cmd=self.preprocess_cmd,
                postprocess_cmd=self.postprocess_cmd,
                wandb_parameters=self.wandb_parameters,
                with_sandbox=self.with_sandbox,
                script=self.script,
                requirements=self.requirements,
                # Multi-model parameters (None for single-model)
                server_addresses=server_addresses,
                model_names=self.model_names,
                server_types=self.server_types,
            )

            # Return command and runtime metadata (environment vars)
            return cmd, {"environment": env_vars}

        # Always use lazy command building
        self.set_inline(build_cmd)
        super().__post_init__()
