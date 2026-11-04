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

"""DeepSWE: SWE-bench agent generation + Harbor verifier grading.

Generation (SWE-agent / mini-SWE-agent / OpenHands / gold_patch) reuses
``SweBenchGenerationTask``. Grading is decoupled: instead of the SWE-bench
harness, each task is scored with its Harbor ``tests/test.sh`` verifier.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys

import hydra

from nemo_skills.inference.eval.harbor_utils import load_verifier_reward
from nemo_skills.inference.eval.swebench import (
    SweBenchGenerationConfig,
    SweBenchGenerationTask,
)
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))

# Re-export for tests / external callers.
_load_verifier_reward = load_verifier_reward


def build_verifier_command(instance_id: str) -> str:
    """Build the DeepSWE v1.1 Harbor verifier command."""
    out_dir = f"/trajectories_mount/eval-outputs/{instance_id}"
    return (
        "mkdir -p /logs/artifacts /logs/verifier && "
        "cp /patch_mount/model.patch /logs/artifacts/model.patch && "
        "export TESTS_DIR=/tests && "
        "export VERIFIER_DIR=/logs/verifier && "
        "export APP_DIR=/app && "
        "export ARTIFACTS_DIR=/logs/artifacts && "
        "cd /app && "
        "bash /tests/test.sh; "
        f"mkdir -p {out_dir} && "
        f"cp -r /logs/verifier/. {out_dir}/ && "
        # DeepSWE's crash sentinel may produce reward.txt=-1 instead of reward.json.
        f"if [ -f {out_dir}/reward.json ] || [ -f {out_dir}/reward.txt ]; then touch {out_dir}/.eval_done; fi"
    )


# These verifiers use fixed localhost ports (Anko, HTTPX, and Testem) or require isolated IPv6 loopback
# behavior (Prometheus). Concurrent runs can otherwise collide or hang when they share the host network
# namespace, so run them with private, network-disabled namespaces.
NETWORK_ISOLATED_VERIFIER_TASKS = frozenset(
    {
        "anko-default-function-arguments",
        "httpx-multipart-response-parsing",
        "httpx-streaming-json-iteration",
        "prometheus-transactional-reload-status",
        "testem-bail-on-test-failure",
        "testem-per-launcher-reports",
    }
)


@nested_dataclass(kw_only=True)
class DeepSweGenerationConfig(SweBenchGenerationConfig):
    # Whether to apply timeouts on the agent and verifier (evaluation stage) if they are set in the data point.
    use_agent_timeouts: bool = False
    use_verifier_timeouts: bool = False


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_deepswe_generation_config", node=DeepSweGenerationConfig)


def parse_deepswe_reward(reward: dict | None, *, patch_exists: bool) -> dict:
    """Normalize Harbor reward.json into NeMo-Skills metrics fields.

    DeepSWE ``tests/test.sh`` writes ``reward.txt=-1`` (EXIT trap) when the verifier
    crashes before producing ``reward.json``; treat negative rewards as infra failure.
    """
    if not patch_exists:
        return {
            "resolved": False,
            "patch_exists": False,
            "patch_successfully_applied": False,
            "reward": 0,
            "f2p": 0.0,
            "p2p": 0.0,
            "partial": 0.0,
        }

    if not isinstance(reward, dict):
        return {
            "resolved": False,
            "patch_exists": True,
            "patch_successfully_applied": False,
            "reward": 0,
            "f2p": 0.0,
            "p2p": 0.0,
            "partial": 0.0,
            "verifier_crashed": True,
        }

    apply_failed = bool(reward.get("apply_failed"))
    raw_reward = reward.get("reward", 0)
    try:
        reward_value = float(raw_reward)
    except (TypeError, ValueError):
        reward_value = 0.0

    # Crash sentinel from DeepSWE test.sh EXIT trap (no reward.json written).
    if reward_value < 0:
        return {
            "resolved": False,
            "patch_exists": True,
            "patch_successfully_applied": False,
            "reward": reward_value,
            "f2p": 0.0,
            "p2p": 0.0,
            "partial": 0.0,
            "verifier_crashed": True,
            "raw_reward": reward,
        }

    return {
        "resolved": (not apply_failed) and reward_value >= 1.0,
        "patch_exists": True,
        "patch_successfully_applied": not apply_failed,
        "reward": reward_value,
        "f2p": float(reward.get("f2p") or 0.0),
        "p2p": float(reward.get("p2p") or 0.0),
        "partial": float(reward.get("partial") or 0.0),
        "f2p_passed": reward.get("f2p_passed"),
        "f2p_total": reward.get("f2p_total"),
        "p2p_passed": reward.get("p2p_passed"),
        "p2p_total": reward.get("p2p_total"),
        "raw_reward": reward,
    }


class DeepSweGenerationTask(SweBenchGenerationTask):
    """SWE-bench agents for generation; Harbor ``tests/test.sh`` for grading."""

    def __init__(self, cfg: DeepSweGenerationConfig):
        if cfg.execution_backend != "opensandbox":
            raise ValueError(
                "DeepSWE v1.1 requires ++execution_backend=opensandbox; "
                "Apptainer container_formatter images are not supported."
            )
        super().__init__(cfg)

    async def _execute_container_command(
        self,
        data_point,
        command,
        expected_file_pattern,
        mode,
        timeout=100000,
        extra_apptainer_args="",
        extra_mounts=(),
    ):
        # Wrapper for the parent class's method that injects per-task agent/verifier timeouts if they are enabled.
        if mode == "agent" and self.cfg.use_agent_timeouts and data_point.get("agent_timeout_sec"):
            timeout = int(data_point.get("agent_timeout_sec")) + 120
        if mode == "eval" and self.cfg.use_verifier_timeouts and data_point.get("verifier_timeout_sec"):
            timeout = int(data_point.get("verifier_timeout_sec")) + 120
        return await SweBenchGenerationTask._execute_container_command(
            self, data_point, command, expected_file_pattern, mode, timeout, extra_apptainer_args, extra_mounts
        )

    async def _run_deepswe_verifier(self, data_point, model_patch: str) -> dict:
        """Apply model.patch in a pristine task image and run tests/test.sh."""
        patches_dir = self.output_dir / "patches"
        patches_dir.mkdir(parents=True, exist_ok=True)
        patch_path = patches_dir / f"{data_point['instance_id']}.patch"
        patch_path.write_text(model_patch if model_patch.endswith("\n") else model_patch + "\n")

        eval_out = self.output_dir / "eval-outputs" / data_point["instance_id"]
        # Drop stale rewards from prior runs/retries so a failed verifier cannot
        # accidentally succeed by matching an old reward.json on disk.
        if eval_out.exists():
            shutil.rmtree(eval_out)
        eval_out.mkdir(parents=True, exist_ok=True)

        # The v1.1 verifier image already contains its pinned tests under /tests.
        # Only the model patch crosses into the pristine verifier sandbox.
        extra_mounts = [(patch_path, "/patch_mount/model.patch", True)]
        extra_apptainer_args = ""
        if self.cfg.execution_backend == "apptainer" and data_point["instance_id"] in NETWORK_ISOLATED_VERIFIER_TASKS:
            # On OpenSandbox, containers never share the same network namespace, so there is no need to isolate.
            extra_apptainer_args = " --net --network none "

        verifier_cmd = build_verifier_command(data_point["instance_id"])

        search_path = str(eval_out / ".eval_done")
        try:
            await self._execute_container_command(
                data_point,
                verifier_cmd,
                search_path,
                mode="eval",
                timeout=self.cfg.swebench_tests_timeout + 120,
                extra_apptainer_args=extra_apptainer_args,
                extra_mounts=extra_mounts,
            )
        except (ValueError, RuntimeError):
            if not (eval_out / "reward.json").exists() and not (eval_out / "reward.txt").exists():
                LOG.error("DeepSWE verifier failed for %s", data_point["instance_id"])
                return parse_deepswe_reward(None, patch_exists=True)

        reward = load_verifier_reward(eval_out)
        return parse_deepswe_reward(reward, patch_exists=True)

    def _get_terminal_error_metrics(self, error: Exception) -> dict:
        """Represent an agent failure as a terminal unresolved DeepSWE trial."""
        metrics = parse_deepswe_reward(None, patch_exists=False)
        metrics["generation_error"] = {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        return metrics

    async def process_single_datapoint(self, data_point, data, prompt_format=None):
        """Will do all necessary generations to get a single answer for the data point."""

        # Run the agent rollout.
        # The semaphore ensures that no more than max_concurrent_requests rollouts are running at the same time.
        async with self.rollout_semaphore:
            pred_file = await self._run_agent(data_point)

        with open(pred_file, "r") as f:
            trajectory_dict = json.loads(f.read())

        model_patch = trajectory_dict["model_patch"]
        has_patch = bool(model_patch and str(model_patch).strip())

        if not has_patch:
            metrics = parse_deepswe_reward(None, patch_exists=False)
        elif not self.cfg.evaluate:
            metrics = {
                "resolved": None,
                "patch_exists": True,
                "patch_successfully_applied": None,
                "reward": None,
                "f2p": None,
                "p2p": None,
                "partial": None,
            }
        else:
            async with self.eval_semaphore:
                metrics = await self._run_deepswe_verifier(data_point, str(model_patch))

        return {
            "swe-bench-metrics": metrics,
            "swe-bench-outputs": trajectory_dict,
            "generation": "",  # required TODO: we should fix this
        }


GENERATION_TASK_CLASS = DeepSweGenerationTask


@hydra.main(version_base=None, config_name="base_deepswe_generation_config")
def deepswe_generation(cfg: DeepSweGenerationConfig):
    cfg = DeepSweGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    DeepSweGenerationTask(cfg).generate()


HELP_MESSAGE = get_help_message(
    DeepSweGenerationConfig,
    server_params=server_params(),
)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        deepswe_generation()
