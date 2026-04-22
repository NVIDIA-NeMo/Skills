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

"""Generation client script for NeMo-Skills pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from nemo_skills.pipeline.utils.generation import get_generation_cmd
from nemo_skills.pipeline.utils.scripts.base import BaseJobScript
from nemo_skills.pipeline.utils.scripts.server import SandboxScript, ServerScript
from nemo_skills.pipeline.utils.server import wrap_python_path


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
        script: Module or file path for generation script
        servers: List of ServerScript references (None for pre-hosted servers)
        server_addresses_prehosted: Addresses for pre-hosted servers
        model_names: Model names for multi-model generation (optional)
        server_types: Server types for multi-model generation (optional)
        sandbox: Reference to SandboxScript (optional)
        log_prefix: Prefix for log files (default: "main")
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
        def build_cmd() -> Tuple[str, Dict]:
            env_vars = {}

            if self.sandbox:
                env_vars["NEMO_SKILLS_SANDBOX_PORT"] = str(self.sandbox.port)

            server_addresses = None
            if self.servers is not None:
                server_addresses = []
                for server_idx, server_script in enumerate(self.servers):
                    if server_script is not None:
                        addr = f"{server_script.hostname_ref()}:{server_script.port}"
                    else:
                        addr = self.server_addresses_prehosted[server_idx]
                    server_addresses.append(addr)

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
                server_addresses=server_addresses,
                model_names=self.model_names,
                server_types=self.server_types,
            )

            return cmd, {"environment": env_vars}

        self.set_inline(build_cmd)
        super().__post_init__()


def _combine_cmds(cmds: List[str], single_node_mode: str) -> str:
    """Combine multiple eval commands into a single shell command.

    Duplicated (intentionally) from pipeline/utils/eval.py to avoid import cycles.
    """
    if single_node_mode == "sequential":
        return " && ".join(cmds)
    if single_node_mode == "parallel":
        if len(cmds) == 1:
            return cmds[0]
        return " & ".join(f"( {cmd} )" for cmd in cmds) + " & wait "
    raise ValueError(f"Unknown single_node_mode: {single_node_mode}")


def _inject_if_missing(extra_arguments: str, needle: str, insertion: str) -> str:
    """Prepend insertion if needle isn't already present."""
    if needle in extra_arguments:
        return extra_arguments
    return f"{insertion}{extra_arguments}"


def _inject_single_server_overrides(
    *,
    extra_arguments: str,
    server_type: str,
    model_name: str,
    host: str | None = None,
    port: int | None = None,
    base_url: str | None = None,
) -> str:
    """Inject single-model server config overrides (like configure_client does).

    We do this at runtime for EvalClientScript so hostname refs can resolve in heterogeneous jobs.
    """
    # Respect user-provided overrides
    extra_arguments = _inject_if_missing(
        extra_arguments, "++server.server_type=", f"++server.server_type={server_type} "
    )
    extra_arguments = _inject_if_missing(extra_arguments, "++server.model=", f"++server.model={model_name} ")

    if host is not None and port is not None:
        extra_arguments = _inject_if_missing(extra_arguments, "++server.host=", f"++server.host={host} ")
        extra_arguments = _inject_if_missing(extra_arguments, "++server.port=", f"++server.port={port} ")
        return extra_arguments

    if base_url is not None:
        extra_arguments = _inject_if_missing(extra_arguments, "++server.base_url=", f"++server.base_url={base_url} ")
        return extra_arguments

    return extra_arguments


@dataclass(kw_only=True)
class EvalClientScript(BaseJobScript):
    """Script for running evaluation generation commands (possibly multiple) in one job.

    Unlike GenerationClientScript (which builds a single generation run),
    this script builds N generation commands and combines them with sequential/parallel mode.

    It supports multi-model server references by resolving a list of server addresses
    at runtime (hostname refs in heterogeneous jobs).
    """

    # A list of generation "units" (dict payload to keep utils->scripts import simple)
    # Each unit must contain args accepted by get_generation_cmd().
    units: List[Dict]
    single_node_mode: str = "parallel"

    # Multi-model cross-component references
    servers: Optional[List[Optional["ServerScript"]]] = None
    server_addresses_prehosted: Optional[List[str]] = None
    model_names: Optional[List[str]] = None
    server_types: Optional[List[str]] = None
    sandbox: Optional["SandboxScript"] = None
    with_sandbox: bool = False

    log_prefix: str = field(default="main", init=False)

    def __post_init__(self):
        def build_cmd() -> Tuple[str, Dict]:
            env_vars: Dict[str, str] = {}

            # Add sandbox port to environment if sandbox is referenced
            if self.sandbox:
                env_vars["NEMO_SKILLS_SANDBOX_PORT"] = str(self.sandbox.port)

            # Build server addresses if servers are provided
            server_addresses = None
            if self.servers is not None:
                server_addresses = []
                for server_idx, server_script in enumerate(self.servers):
                    if server_script is not None:
                        addr = f"{server_script.hostname_ref()}:{server_script.port}"
                    else:
                        addr = self.server_addresses_prehosted[server_idx]
                    server_addresses.append(addr)

            cmds: List[str] = []
            is_multi_model = bool(self.model_names) and len(self.model_names) > 1
            for unit in self.units:
                unit = dict(unit)

                # Single-model: get_generation_cmd doesn't emit list-style ++server.* overrides,
                # so we must inject the single-server config here (mirrors configure_client behavior).
                if self.model_names and len(self.model_names) == 1:
                    server_type = self.server_types[0] if self.server_types else ""
                    model_name = self.model_names[0]
                    if self.servers is not None and self.servers[0] is not None:
                        srv = self.servers[0]
                        unit["extra_arguments"] = _inject_single_server_overrides(
                            extra_arguments=unit.get("extra_arguments", ""),
                            server_type=server_type,
                            model_name=model_name,
                            host=srv.hostname_ref(),
                            port=srv.port,
                        )
                    else:
                        unit["extra_arguments"] = _inject_single_server_overrides(
                            extra_arguments=unit.get("extra_arguments", ""),
                            server_type=server_type,
                            model_name=model_name,
                            base_url=server_addresses[0] if server_addresses else None,
                        )

                cmds.append(
                    get_generation_cmd(
                        **unit,
                        # Multi-model parameters (only for num_models > 1)
                        server_addresses=server_addresses if is_multi_model else None,
                        model_names=self.model_names if is_multi_model else None,
                        server_types=self.server_types if is_multi_model else None,
                    )
                )

            combined = _combine_cmds(cmds, self.single_node_mode)
            combined = wrap_python_path(combined)
            return combined, {"environment": env_vars}

        self.set_inline(build_cmd)
        super().__post_init__()
