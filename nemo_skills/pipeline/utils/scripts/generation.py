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
