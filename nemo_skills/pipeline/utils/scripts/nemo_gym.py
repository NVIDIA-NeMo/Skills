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

"""NeMo Gym rollout collection script for NeMo-Skills pipeline."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from nemo_skills.pipeline.utils.scripts.base import BaseJobScript
from nemo_skills.pipeline.utils.scripts.server import SandboxScript, ServerScript
from nemo_skills.utils import get_server_wait_cmd


@dataclass(kw_only=True)
class NemoGymRolloutsScript(BaseJobScript):
    """Script for running NeMo Gym rollout collection.

    This script orchestrates the full rollout collection workflow:
    1. Starts ng_run in background to spin up NeMo Gym servers
    2. Polls ng_status until all servers are healthy
    3. Runs ng_collect_rollouts to collect rollouts
    4. Keeps ng_run running (cleanup handled externally)

    Attributes:
        config_paths: List of YAML config file paths for ng_run
        input_file: Input JSONL file path for rollout collection
        output_file: Output JSONL file path for rollouts
        extra_arguments: Additional Hydra overrides passed to both ng_run and ng_collect_rollouts
        server: Optional ServerScript reference for policy model server
        server_address: Optional pre-hosted server address
        sandbox: Optional SandboxScript reference for sandbox port
        gym_path: Path to NeMo Gym installation
        policy_api_key: API key for policy server
        policy_model_name: Model name override for policy server
        log_prefix: Prefix for log files (default: "nemo_gym")
    """

    config_paths: List[str]
    input_file: str
    output_file: str
    extra_arguments: str = ""
    server: Optional["ServerScript"] = None
    server_address: Optional[str] = None
    sandbox: Optional["SandboxScript"] = None
    gym_path: str = "/opt/NeMo-RL/3rdparty/Gym-workspace/Gym"
    policy_api_key: str = "dummy"
    policy_model_name: Optional[str] = None

    log_prefix: str = field(default="nemo_gym", init=False)

    def __post_init__(self):
        """Initialize the combined ng_run + ng_collect_rollouts script."""
        if self.server is not None and self.server_address is not None:
            raise ValueError("Specify only one of `server` or `server_address`.")

        def build_cmd() -> Tuple[str, Dict]:
            """Build the full rollout collection command."""
            config_paths_str = ",".join(self.config_paths)

            # Build ng_run command parts
            ng_run_parts = [
                "ng_run",
                f'"+config_paths=[{config_paths_str}]"',
            ]

            if self.server is not None:
                server_addr = f"http://{self.server.hostname_ref()}:{self.server.port}/v1"
                ng_run_parts.append(f'+policy_base_url="{server_addr}"')
            elif self.server_address is not None:
                ng_run_parts.append(f'+policy_base_url="{self.server_address}"')

            ng_run_parts.append(f'+policy_api_key="{self.policy_api_key}"')

            if self.policy_model_name:
                ng_run_parts.append(f'+policy_model_name="{self.policy_model_name}"')

            if self.extra_arguments:
                ng_run_parts.append(self.extra_arguments)

            ng_run_cmd = " ".join(ng_run_parts)

            # Build ng_collect_rollouts command
            ng_collect_parts = [
                "ng_collect_rollouts",
                f'+input_jsonl_fpath="{self.input_file}"',
                f'+output_jsonl_fpath="{self.output_file}"',
            ]

            if self.extra_arguments:
                ng_collect_parts.append(self.extra_arguments)

            ng_collect_cmd = " ".join(ng_collect_parts)

            # Compute the vLLM server URL for the wait check
            if self.server is not None:
                vllm_server_url = f"http://{self.server.hostname_ref()}:{self.server.port}/v1"
            elif self.server_address is not None:
                vllm_server_url = self.server_address
            else:
                vllm_server_url = ""

            # Build server wait command using shared utility
            if vllm_server_url:
                server_wait_cmd = get_server_wait_cmd(f"{vllm_server_url}/models")
            else:
                server_wait_cmd = ""

            cmd = f"""set -e
set -o pipefail

# Install/sync NeMo Gym venv. The nemo-rl container has Gym pre-installed,
# but when users mount a custom Gym path (e.g., from a dev branch or worktree),
# the mounted directory may not have a .venv. The --allow-existing flag makes
# this fast (~1s) when the venv already exists and is up to date.
echo "=== Installing NeMo Gym ==="
cd {self.gym_path} || {{ echo "ERROR: Failed to cd to Gym directory"; exit 1; }}
uv venv --python 3.12 --allow-existing .venv || {{ echo "ERROR: Failed to create venv"; exit 1; }}
source .venv/bin/activate || {{ echo "ERROR: Failed to activate venv"; exit 1; }}
uv sync --active --extra dev || {{ echo "ERROR: Failed to sync dependencies"; exit 1; }}
echo "NeMo Gym installed successfully"

# Disable pipefail for the polling loop (grep may return non-zero)
set +o pipefail

# Wait for vLLM server to be ready before starting ng_run
# Note: --kill-on-bad-exit in srun ensures job fails if vLLM crashes
if [ -n "{vllm_server_url}" ]; then
    echo "=== Waiting for vLLM server at {vllm_server_url} ==="
    {server_wait_cmd}
    echo "vLLM server is ready!"
fi

echo "=== Starting NeMo Gym servers ==="
{ng_run_cmd} &
NG_RUN_PID=$!
echo "ng_run PID: $NG_RUN_PID"

echo "Waiting for NeMo Gym servers..."
LAST_STATUS=""
while true; do
    # Check if ng_run process died - let the failure cascade naturally
    if ! kill -0 $NG_RUN_PID 2>/dev/null; then
        echo "ERROR: ng_run process exited unexpectedly"
        wait $NG_RUN_PID 2>/dev/null  # Get exit code
        exit 1
    fi

    STATUS_OUTPUT=$(ng_status 2>&1)

    if echo "$STATUS_OUTPUT" | grep -q "healthy, 0 unhealthy"; then
        echo "All servers ready!"
        break
    fi

    # Only print status when it changes (reduce verbosity)
    CURRENT_STATUS=$(echo "$STATUS_OUTPUT" | grep -oE '[0-9]+ healthy' | head -1 || echo "starting")
    if [ "$CURRENT_STATUS" != "$LAST_STATUS" ]; then
        echo "Server status: $CURRENT_STATUS"
        LAST_STATUS="$CURRENT_STATUS"
    fi

    sleep 10
done

# Re-enable pipefail for the actual rollout collection
set -o pipefail

echo "=== Running rollout collection ==="
echo "Input file: {self.input_file}"
echo "Output file: {self.output_file}"
mkdir -p "$(dirname "{self.output_file}")"
echo "Output directory created: $(dirname "{self.output_file}")"
echo "Running: {ng_collect_cmd}"
{ng_collect_cmd} || {{ echo "ERROR: ng_collect_rollouts failed"; kill $NG_RUN_PID 2>/dev/null || true; exit 1; }}

echo "=== Rollout collection complete ==="
echo "Output: {self.output_file}"

echo "=== Cleaning up ==="
kill $NG_RUN_PID 2>/dev/null || true
echo "Servers terminated."
"""
            env_vars = {}
            if self.sandbox is not None:
                env_vars["NEMO_SKILLS_SANDBOX_HOST"] = self.sandbox.hostname_ref()
                env_vars["NEMO_SKILLS_SANDBOX_PORT"] = str(self.sandbox.port)

            return cmd.strip(), {"environment": env_vars}

        self.set_inline(build_cmd)
        super().__post_init__()
