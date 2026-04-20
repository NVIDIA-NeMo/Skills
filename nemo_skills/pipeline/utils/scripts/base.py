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

"""Base script class for NeMo-Skills pipeline components."""

from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import nemo_run as run

from nemo_skills.pipeline.utils.exp import install_packages_wrap


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
        num_tasks_override: Override the group's num_tasks for this specific script.
            When set, this script's srun will use this value for --ntasks-per-node
            instead of the group's HardwareConfig.num_tasks. Useful when multiple
            scripts in a CommandGroup need different task configurations (e.g.,
            vLLM servers needing 2 tasks per node while Gym client needs 1).
    """

    het_group_index: Optional[int] = field(default=None, init=False, repr=False)
    span_group_nodes: bool = False  # Default: run on 1 node
    num_tasks_override: Optional[int] = None  # Per-script task count override
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
