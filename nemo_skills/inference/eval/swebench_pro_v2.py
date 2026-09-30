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

"""SWE-bench Pro V2 generation with fresh-image Harbor grading."""

from __future__ import annotations

import json
import logging
import shutil
import sys
from pathlib import Path

import hydra

from nemo_skills.inference.eval.harbor_utils import load_verifier_reward, resolve_tests_dir
from nemo_skills.inference.eval.swebench import SweBenchGenerationConfig, SweBenchGenerationTask
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


def _resolve_tests_dir(data_point: dict, eval_config: dict | None = None, *, tasks_dir: str | None = None):
    return resolve_tests_dir(
        data_point,
        eval_config,
        tasks_dir=tasks_dir,
        benchmark_name="swe-bench-pro-v2",
    )


@nested_dataclass(kw_only=True)
class SweBenchProV2GenerationConfig(SweBenchGenerationConfig):
    isolate_agent_network: bool = True
    use_agent_timeouts: bool = True
    use_verifier_timeouts: bool = True
    tasks_dir: str | None = None


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_swebench_pro_v2_generation_config", node=SweBenchProV2GenerationConfig)


def parse_swebench_pro_v2_reward(reward: dict | None, *, patch_exists: bool) -> dict:
    if not patch_exists:
        return {
            "resolved": False,
            "patch_exists": False,
            "patch_successfully_applied": False,
            "reward": 0,
        }

    if isinstance(reward, dict) and reward.get("apply_failed"):
        return {
            "resolved": False,
            "patch_exists": True,
            "patch_successfully_applied": False,
            "reward": 0,
            "apply_error": reward.get("apply_error"),
        }

    if not isinstance(reward, dict):
        return {
            "resolved": False,
            "patch_exists": True,
            "patch_successfully_applied": True,
            "reward": 0,
            "verifier_error": True,
        }

    try:
        reward_value = float(reward.get("reward", 0))
    except (TypeError, ValueError):
        reward_value = 0.0
    return {
        "resolved": reward_value >= 1.0,
        "patch_exists": True,
        "patch_successfully_applied": True,
        "reward": reward_value,
        "raw_reward": reward,
    }


class SweBenchProV2GenerationTask(SweBenchGenerationTask):
    """Run standard SWE agents and grade their diffs in pristine V2 images."""

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
        if mode == "agent" and self.cfg.use_agent_timeouts and data_point.get("agent_timeout_sec"):
            timeout = int(data_point["agent_timeout_sec"]) + 120
        if mode == "eval" and self.cfg.use_verifier_timeouts and data_point.get("verifier_timeout_sec"):
            timeout = int(data_point["verifier_timeout_sec"]) + 120
        return await SweBenchGenerationTask._execute_container_command(
            self,
            data_point,
            command,
            expected_file_pattern,
            mode,
            timeout,
            extra_apptainer_args,
            extra_mounts,
        )

    def _resolve_tests_dir(self, data_point: dict) -> Path:
        return resolve_tests_dir(
            data_point,
            self.cfg.eval_config,
            tasks_dir=self.cfg.tasks_dir,
            benchmark_name="swe-bench-pro-v2",
        )

    async def _run_swebench_pro_v2_verifier(self, data_point: dict, model_patch: str) -> dict:
        instance_id = data_point["instance_id"]
        patches_dir = self.output_dir / "patches"
        patches_dir.mkdir(parents=True, exist_ok=True)
        patch_path = patches_dir / f"{instance_id}.patch"
        patch_path.write_text(model_patch if model_patch.endswith("\n") else model_patch + "\n")

        eval_out = self.output_dir / "eval-outputs" / instance_id
        if eval_out.exists():
            shutil.rmtree(eval_out)
        eval_out.mkdir(parents=True)

        extra_mounts = [
            (self._resolve_tests_dir(data_point), "/tests", True),
            (patch_path, "/patch_mount/model.patch", True),
        ]
        # This mirrors v2/tooling/patch_replay.py: only the diff enters a fresh
        # task image, then the unchanged Harbor verifier grades that tree.
        verifier_cmd = (
            "mkdir -p /logs/artifacts /logs/verifier && "
            "cp /patch_mount/model.patch /logs/artifacts/model.patch && "
            "repo=/app; [ -d /app/.git ] || repo=/testbed; "
            'if ! cd "$repo"; then '
            '  echo \'{"reward":0,"apply_failed":1,"apply_error":"repository not found"}\' '
            "    > /logs/verifier/reward.json; "
            "elif ! (git apply --verbose /logs/artifacts/model.patch "
            "    || git apply --3way /logs/artifacts/model.patch "
            "    || patch --fuzz=3 -p1 -i /logs/artifacts/model.patch); then "
            '  echo \'{"reward":0,"apply_failed":1,"apply_error":"model patch did not apply"}\' '
            "    > /logs/verifier/reward.json; "
            "else "
            "  export TESTS_DIR=/tests VERIFIER_DIR=/logs/verifier ARTIFACTS_DIR=/logs/artifacts; "
            '  export APP_DIR="$repo"; '
            "  bash /tests/test.sh || true; "
            "fi; "
            f"mkdir -p /trajectories_mount/eval-outputs/{instance_id} && "
            f"cp -r /logs/verifier/. /trajectories_mount/eval-outputs/{instance_id}/"
        )

        search_path = str(eval_out / "reward.*")
        try:
            await self._execute_container_command(
                data_point,
                verifier_cmd,
                search_path,
                mode="eval",
                timeout=self.cfg.swebench_tests_timeout + 120,
                extra_mounts=extra_mounts,
            )
        except ValueError:
            if not (eval_out / "reward.json").exists() and not (eval_out / "reward.txt").exists():
                LOG.error("SWE-bench Pro V2 verifier failed for %s", instance_id)
                return parse_swebench_pro_v2_reward(None, patch_exists=True)

        return parse_swebench_pro_v2_reward(load_verifier_reward(eval_out), patch_exists=True)

    def _get_terminal_error_metrics(self, error: Exception) -> dict:
        metrics = parse_swebench_pro_v2_reward(None, patch_exists=False)
        metrics["generation_error"] = {
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
        return metrics

    async def process_single_datapoint(self, data_point, data, prompt_format=None):
        async with self.rollout_semaphore:
            pred_file = await self._run_agent(data_point)

        with open(pred_file) as fin:
            trajectory_dict = json.load(fin)
        model_patch = trajectory_dict["model_patch"]
        has_patch = bool(model_patch and str(model_patch).strip())

        if not has_patch:
            metrics = parse_swebench_pro_v2_reward(None, patch_exists=False)
        elif not self.cfg.evaluate:
            metrics = {
                "resolved": None,
                "patch_exists": True,
                "patch_successfully_applied": None,
                "reward": None,
            }
        else:
            async with self.eval_semaphore:
                metrics = await self._run_swebench_pro_v2_verifier(data_point, str(model_patch))

        return {
            "swe-bench-metrics": metrics,
            "swe-bench-outputs": trajectory_dict,
            "generation": "",
        }


GENERATION_TASK_CLASS = SweBenchProV2GenerationTask


@hydra.main(version_base=None, config_name="base_swebench_pro_v2_generation_config")
def swebench_pro_v2_generation(cfg: SweBenchProV2GenerationConfig):
    cfg = SweBenchProV2GenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    SweBenchProV2GenerationTask(cfg).generate()


HELP_MESSAGE = get_help_message(SweBenchProV2GenerationConfig, server_params=server_params())


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        swebench_pro_v2_generation()
