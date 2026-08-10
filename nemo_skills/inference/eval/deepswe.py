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
from pathlib import Path

import hydra

from nemo_skills.inference.eval.swebench import (
    SweBenchGenerationConfig,
    SweBenchGenerationTask,
)
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))

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

    # Custom path to the Harbor tasks root (contains <instance_id>/tests/).
    # Defaults to <data_dir>/deep-swe/tasks.
    tasks_dir: str | None = None


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


def _load_verifier_reward(eval_out: Path) -> dict | None:
    reward_json = eval_out / "reward.json"
    if reward_json.exists():
        try:
            return json.loads(reward_json.read_text())
        except json.JSONDecodeError:
            LOG.warning("Invalid reward.json at %s", reward_json)
            return None

    reward_txt = eval_out / "reward.txt"
    if reward_txt.exists():
        raw = reward_txt.read_text().strip()
        try:
            return {"reward": float(raw)}
        except ValueError:
            return {"reward": 0}
    return None


class DeepSweGenerationTask(SweBenchGenerationTask):
    """SWE-bench agents for generation; Harbor ``tests/test.sh`` for grading."""

    async def _execute_container_command(
        self,
        data_point,
        command,
        expected_file_pattern,
        mode,
        timeout=100000,
        extra_apptainer_args="",
    ):
        # Wrapper for the parent class's method that injects per-task agent/verifier timeouts if they are enabled.
        if mode == "agent" and self.cfg.use_agent_timeouts and data_point.get("agent_timeout_sec"):
            timeout = int(data_point.get("agent_timeout_sec")) + 120
        if mode == "eval" and self.cfg.use_verifier_timeouts and data_point.get("verifier_timeout_sec"):
            timeout = int(data_point.get("verifier_timeout_sec")) + 120
        return await SweBenchGenerationTask._execute_container_command(
            self, data_point, command, expected_file_pattern, mode, timeout, extra_apptainer_args
        )

    def _resolve_tests_dir(self, data_point: dict) -> Path:
        """Resolve Harbor tests/ for one task.

        Preference order:
        1. ``++tasks_dir``/<instance_id>/tests (explicit Harbor tasks root override if passed)
        2. ``++eval_config.data_dir``/deep-swe/tasks/<instance_id>/tests (set automatically by ``ns eval --data_dir``)
        """
        instance_id = data_point["instance_id"]

        tasks_dir = self.cfg.tasks_dir
        if tasks_dir:
            candidate = Path(tasks_dir) / instance_id / "tests"
            if candidate.is_dir():
                return candidate
            raise ValueError(
                f"DeepSWE tests dir not found for {instance_id}: {candidate}. "
                f"Check ++tasks_dir={tasks_dir} (expected <tasks_dir>/<instance_id>/tests)."
            )

        # ns eval --data_dir=... injects ++eval_config.data_dir
        data_dir = self.cfg.eval_config.get("data_dir")
        if data_dir:
            candidate = Path(data_dir) / "deep-swe" / "tasks" / instance_id / "tests"
            if candidate.is_dir():
                return candidate
            raise ValueError(
                f"DeepSWE tests dir not found for {instance_id}: {candidate}. "
                f"Expected tasks under {{data_dir}}/deep-swe/tasks after "
                f"`ns prepare_data deep-swe --data_dir=...`. "
                f"Or pass ++tasks_dir explicitly."
            )

        raise ValueError(
            f"DeepSWE tests dir not found for {instance_id}. "
            "Pass --data_dir (recommended) or ++tasks_dir=/path/to/deep-swe/tasks "
            "after ns prepare_data."
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

        tests_dir = self._resolve_tests_dir(data_point)
        extra_apptainer_args = f" --mount type=bind,src={tests_dir},dst=/tests,ro "
        if data_point["instance_id"] in NETWORK_ISOLATED_VERIFIER_TASKS:
            extra_apptainer_args += " --net --network none "

        verifier_cmd = (
            "mkdir -p /logs/artifacts /logs/verifier && "
            f"cp /trajectories_mount/patches/{data_point['instance_id']}.patch /logs/artifacts/model.patch && "
            "export TESTS_DIR=/tests && "
            "export VERIFIER_DIR=/logs/verifier && "
            "export APP_DIR=/app && "
            "export ARTIFACTS_DIR=/logs/artifacts && "
            "cd /app && "
            "bash /tests/test.sh; "
            f"mkdir -p /trajectories_mount/eval-outputs/{data_point['instance_id']} && "
            f"cp -r /logs/verifier/. /trajectories_mount/eval-outputs/{data_point['instance_id']}/"
        )

        # Prefer reward.json; DeepSWE crash sentinel only writes reward.txt=-1.
        search_path = str(eval_out / "reward.*")
        try:
            await self._execute_container_command(
                data_point,
                verifier_cmd,
                search_path,
                mode="eval",
                timeout=self.cfg.swebench_tests_timeout + 120,
                extra_apptainer_args=extra_apptainer_args,
            )
        except ValueError:
            if not (eval_out / "reward.json").exists() and not (eval_out / "reward.txt").exists():
                LOG.error("DeepSWE verifier failed for %s", data_point["instance_id"])
                return parse_deepswe_reward(None, patch_exists=True)

        reward = _load_verifier_reward(eval_out)
        return parse_deepswe_reward(reward, patch_exists=True)

    async def process_single_datapoint(self, data_point, data, prompt_format=None):
        """Will do all necessary generations to get a single answer for the data point."""

        # Run the agent rollout.
        # The semaphore ensures that no more than max_concurrent_requests rollouts are running at the same time.
        async with self.semaphore:
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
            async with self.semaphore:
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
