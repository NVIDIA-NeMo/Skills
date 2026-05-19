# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""`GymEvalClientScript` — peer of `EvalClientScript` for the Gym backend.

Builds a single shell command that:
1. Resolves the Gym install path (importable `nemo_gym` package → fall back to container).
2. Activates the Gym venv (`uv sync --active --extra dev`).
3. Waits for the policy vLLM server to become ready.
4. Starts `ng_run` in the background with the benchmark's `config_paths`.
5. Polls `ng_status` until all resource/agent servers are healthy.
6. Runs `ng_collect_rollouts` once per `EvalGenerationUnit` (each unit has its
   own input/output and per-seed extra_arguments).
7. Cleans up `ng_run`.

`ng_run` is shared across all units in the job (one Gym mesh per SLURM job).
This means a single GymEvalClientScript instance can only handle units that
share the same `config_paths` + `agent_name`; the caller (ns eval dispatcher)
enforces one-benchmark-per-job for the Gym backend.

Sandbox: per the Q5 decision, Skills continues to launch the sandbox; we just
forward its host/port to the Gym mesh through env vars, same as
`NemoGymRolloutsScript` does.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from nemo_skills.pipeline.utils.gym.translator import translate_skills_overrides_to_gym
from nemo_skills.pipeline.utils.scripts.base import BaseJobScript
from nemo_skills.pipeline.utils.scripts.server import SandboxScript, ServerScript
from nemo_skills.utils import get_server_wait_cmd

DEFAULT_GYM_PATH = "/opt/Gym"


def _resolve_gym_path_snippet(gym_path: Optional[str]) -> str:
    """Same scheme as NemoGymRolloutsScript: prefer the importable package, fall back to container."""
    if gym_path is not None:
        return f"GYM_PATH={shlex.quote(str(gym_path))}"
    return f"""GYM_PATH=$(python - <<'PY'
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("nemo_gym")
if spec is not None:
    for candidate in Path(spec.origin).resolve().parents:
        if (candidate / "pyproject.toml").is_file():
            print(candidate)
            raise SystemExit(0)

print({DEFAULT_GYM_PATH!r})
PY
)"""


def _output_jsonl_for_unit(unit: Dict) -> str:
    """Derive the rollouts output file from a Skills EvalGenerationUnit.

    Skills units carry `output_dir` (where output.jsonl is expected) and a
    `random_seed` (None for greedy / single-seed runs, int for multi-sample).
    """
    output_dir = unit["output_dir"]
    seed = unit.get("random_seed")
    chunk_id = unit.get("chunk_id")
    parts = ["rollouts"]
    if seed is not None:
        parts.append(f"rs{seed}")
    if chunk_id is not None:
        parts.append(f"chunk{chunk_id}")
    return f"{output_dir}/{'-'.join(parts)}.jsonl"


@dataclass(kw_only=True)
class GymEvalClientScript(BaseJobScript):
    """Client script for `ns eval --backend=gym`.

    Surface mirrors `EvalClientScript` so the dispatcher can swap them
    symmetrically. Gym-specific fields are `config_paths`, `agent_name`,
    `gym_path`, `policy_api_key`, `policy_model_name`.

    Note on `single_node_mode`: parallelism across units is handled by
    `+num_samples_in_parallel` inside `ng_collect_rollouts` (units run
    sequentially against one shared Gym mesh). We accept the field for
    interface symmetry but ignore it.
    """

    units: List[Dict]
    config_paths: List[str]
    agent_name: str

    # When set, the script rewrites each unit's `input_file` (which comes
    # from Skills' dataset path) to the Gym-shape input file. The value is
    # interpreted relative to the resolved Gym install dir at runtime
    # (typically /opt/Gym). When None, units' input_file is used as-is —
    # only correct when Skills' input schema happens to match what the
    # resource server's `apply_prompt_to_row` expects.
    gym_input_jsonl_fpath: Optional[str] = None

    # When set, passes `+prompt_config=$GYM_PATH/<value>` to ng_collect_rollouts
    # so it renders responses_create_params.input from the Gym prompt YAML.
    # Required for the Skills-derived input JSONLs (which carry raw fields
    # like `question`/`expected_answer`, not pre-built responses_create_params).
    gym_prompt_config: Optional[str] = None

    servers: Optional[List[Optional["ServerScript"]]] = None
    server_addresses_prehosted: Optional[List[str]] = None
    model_names: Optional[List[str]] = None
    server_types: Optional[List[str]] = None
    sandbox: Optional["SandboxScript"] = None

    gym_path: Optional[str] = None
    policy_api_key: str = "dummy"
    policy_model_name: Optional[str] = None

    # Accepted for symmetry with EvalClientScript; unused (see class docstring).
    single_node_mode: str = "parallel"
    with_sandbox: bool = False
    installation_command: Optional[str] = None

    log_prefix: str = field(default="gym_eval", init=False)

    def __post_init__(self):
        if not self.units:
            raise ValueError("GymEvalClientScript requires at least one unit.")
        if not self.config_paths:
            raise ValueError("GymEvalClientScript requires config_paths (Gym yaml).")
        if not self.agent_name:
            raise ValueError("GymEvalClientScript requires agent_name.")
        if self.servers is not None and len(self.servers) > 1:
            raise NotImplementedError(
                "Multi-model evaluation is not supported by the Gym backend in v1. "
                "Use --backend=skills for multi-model evals."
            )

        def build_cmd() -> Tuple[str, Dict]:
            policy_base_url = self._policy_base_url()
            policy_model_name = self.policy_model_name or (self.model_names[0] if self.model_names else None)

            ng_run_cmd = self._ng_run_cmd(policy_base_url)
            # One ng_collect_rollouts per unit. Gym writes `rollouts.jsonl` +
            # `rollouts_aggregate_metrics.json` natively — no post-rollout
            # conversion to Skills schema (we intentionally break the Skills
            # output contract as part of this migration).
            per_unit_steps: List[List[str]] = [[self._ng_collect_cmd(unit, policy_model_name)] for unit in self.units]

            wait_cmd = get_server_wait_cmd(f"{policy_base_url}/models") if policy_base_url else ""
            gym_path_snippet = _resolve_gym_path_snippet(self.gym_path)

            cmd = self._shell_scaffold(
                gym_path_snippet=gym_path_snippet,
                wait_cmd=wait_cmd,
                policy_base_url=policy_base_url,
                ng_run_cmd=ng_run_cmd,
                per_unit_steps=per_unit_steps,
            )

            env_vars: Dict[str, str] = {}
            if self.sandbox is not None:
                env_vars["NEMO_SKILLS_SANDBOX_HOST"] = self.sandbox.hostname_ref()
                env_vars["NEMO_SKILLS_SANDBOX_PORT"] = str(self.sandbox.port)
            return cmd.strip(), {"environment": env_vars}

        self.set_inline(build_cmd)
        super().__post_init__()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _policy_base_url(self) -> str:
        """Resolve the vLLM/OpenAI policy server URL from servers/prehosted."""
        if self.servers and self.servers[0] is not None:
            srv = self.servers[0]
            return f"http://{srv.hostname_ref()}:{srv.port}/v1"
        if self.server_addresses_prehosted:
            return self.server_addresses_prehosted[0]
        return ""

    def _ng_run_cmd(self, policy_base_url: str) -> str:
        config_paths_str = ",".join(self.config_paths)
        parts = ["ng_run", f'"+config_paths=[{config_paths_str}]"']
        if policy_base_url:
            parts.append(f'+policy_base_url="{policy_base_url}"')
        parts.append(f'+policy_api_key="{self.policy_api_key}"')
        if self.policy_model_name:
            parts.append(f'+policy_model_name="{self.policy_model_name}"')
        elif self.model_names:
            parts.append(f'+policy_model_name="{self.model_names[0]}"')
        return " ".join(parts)

    def _ng_collect_cmd(self, unit: Dict, policy_model_name: Optional[str]) -> str:
        if self.gym_input_jsonl_fpath:
            # Resolve at runtime so the path is right regardless of where Gym
            # actually ended up (packaged /nemo_run/code, mounted /opt/Gym,
            # container default). $GYM_PATH is set by the shell scaffold.
            input_file = f'"$GYM_PATH"/{self.gym_input_jsonl_fpath}'
        else:
            input_file = unit["input_file"]
        output_file = _output_jsonl_for_unit(unit)
        translated = translate_skills_overrides_to_gym(unit.get("extra_arguments", ""))

        # input_file may already include shell quoting (when it's an
        # interpolated $GYM_PATH path); only wrap in quotes when it's a
        # plain absolute path.
        quoted_input = input_file if input_file.startswith('"') else f'"{input_file}"'
        parts = [
            "ng_collect_rollouts",
            f"+agent_name={self.agent_name}",
            f"+input_jsonl_fpath={quoted_input}",
            f'+output_jsonl_fpath="{output_file}"',
        ]
        if self.gym_prompt_config:
            parts.append(f'+prompt_config="$GYM_PATH"/{self.gym_prompt_config}')
        # Note: Skills' per-job `random_seed` has no direct equivalent on Gym's
        # responses_create_params (schema is extra='forbid', no `seed` field).
        # The proper Gym mechanism is `+num_repeats_add_seed=true` with
        # `num_repeats>1`, or overriding the model-server's `vllm_model.extra_body`
        # at ng_run time. Per-call seeding intentionally not threaded here; the
        # mean@N statistics still match within noise for the parity pilot.

        if translated:
            parts.append(translated)
        ng_collect = " ".join(parts)
        # ng_collect_rollouts writes `<stem>_materialized_inputs.jsonl` alongside
        # the output JSONL; the parent dir must exist before it runs.
        return f'mkdir -p "$(dirname "{output_file}")" && {ng_collect}'

    def _shell_scaffold(
        self,
        *,
        gym_path_snippet: str,
        wait_cmd: str,
        policy_base_url: str,
        ng_run_cmd: str,
        per_unit_steps: List[List[str]],
    ) -> str:
        unit_blocks: List[str] = []
        for i, steps in enumerate(per_unit_steps):
            step_lines = [f'echo "--- unit {i + 1}/{len(per_unit_steps)} ---"']
            for step in steps:
                step_lines.append(
                    f'{step} || {{ echo "ERROR: ng_collect_rollouts failed (unit {i + 1})"; '
                    f"kill $NG_RUN_PID 2>/dev/null || true; exit 1; }}"
                )
            unit_blocks.append("\n".join(step_lines))
        unit_invocations = "\n".join(unit_blocks)
        wait_block = (
            f'echo "=== Waiting for policy server at {policy_base_url} ==="\n{wait_cmd}\n'
            f'echo "policy server is ready!"'
            if wait_cmd
            else ""
        )
        return f"""set -e
set -o pipefail

echo "=== Installing NeMo Gym ==="
{gym_path_snippet}
echo "Using NeMo Gym path: $GYM_PATH"
cd "$GYM_PATH" || {{ echo "ERROR: Failed to cd to Gym directory: $GYM_PATH"; exit 1; }}
uv venv --python 3.12 --allow-existing .venv || {{ echo "ERROR: Failed to create venv"; exit 1; }}
source .venv/bin/activate || {{ echo "ERROR: Failed to activate venv"; exit 1; }}
uv sync --active --extra dev || {{ echo "ERROR: Failed to sync dependencies"; exit 1; }}

set +o pipefail
{wait_block}

echo "=== Starting NeMo Gym servers ==="
{ng_run_cmd} &
NG_RUN_PID=$!
echo "ng_run PID: $NG_RUN_PID"

LAST_STATUS=""
while true; do
    if ! kill -0 $NG_RUN_PID 2>/dev/null; then
        echo "ERROR: ng_run process exited unexpectedly"
        wait $NG_RUN_PID 2>/dev/null
        exit 1
    fi
    STATUS_OUTPUT=$(ng_status 2>&1)
    if echo "$STATUS_OUTPUT" | grep -q "healthy, 0 unhealthy"; then
        echo "All Gym servers ready!"
        break
    fi
    CURRENT_STATUS=$(echo "$STATUS_OUTPUT" | grep -oE '[0-9]+ healthy' | head -1 || echo "starting")
    if [ "$CURRENT_STATUS" != "$LAST_STATUS" ]; then
        echo "Server status: $CURRENT_STATUS"
        LAST_STATUS="$CURRENT_STATUS"
    fi
    sleep 10
done

set -o pipefail

echo "=== Running rollout collection ({len(per_unit_steps)} unit(s)) ==="
{unit_invocations}

echo "=== Rollout collection complete ==="

echo "=== Cleaning up ==="
kill $NG_RUN_PID 2>/dev/null || true
echo "Done."
"""
