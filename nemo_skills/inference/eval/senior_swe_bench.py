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

"""Senior SWE-Bench: SWE-bench agent generation + Harbor verifier grading.

Generation (SWE-agent / mini-SWE-agent / OpenHands / gold_patch) reuses
``SweBenchGenerationTask``. Grading is decoupled: instead of the SWE-bench
harness, each task is scored with its Harbor ``tests/test.sh`` verifier
(verify + LLM judge + validation stages).

Unlike DeepSWE, Senior SWE-Bench ``test.sh`` expects agent edits already present
under ``/repo/$REPO_NAME``. NeMo-Skills therefore applies ``model.patch`` into
that tree before invoking the verifier. Judge stages need outbound network and
API keys (injected via Apptainer ``--env`` because the harness uses ``--cleanenv``).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import shutil
import sys
from pathlib import Path

import hydra

from nemo_skills.inference.eval.harbor_utils import load_verifier_reward, resolve_tests_dir
from nemo_skills.inference.eval.swebench import (
    SweBenchGenerationConfig,
    SweBenchGenerationTask,
)
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))

# Env vars forwarded into the verifier container for LLM judges / validation.
SSB_VERIFIER_ENV_KEYS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "PORTKEY_API_KEY",
    "SSB_OVERRIDE_ALL_JUDGE_MODEL",
    "SSB_OVERRIDE_CLASSIFIER_MODEL",
    "SSB_OVERRIDE_VA_HARNESS",
    "SSB_OVERRIDE_VA_MODEL",
    "FORCE_VA_HARNESS",
    "FORCE_VA_MODEL",
    "TIMEOUT_MULTIPLIER",
)

_REPO_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_repo_name(repo_name: str) -> str:
    if not _REPO_NAME_RE.fullmatch(repo_name):
        raise ValueError(f"Unsafe REPO_NAME for shell interpolation: {repo_name!r}")
    return repo_name


@nested_dataclass(kw_only=True)
class SeniorSweBenchGenerationConfig(SweBenchGenerationConfig):
    # Whether to apply timeouts on the agent and verifier if set on the data point.
    use_agent_timeouts: bool = False
    use_verifier_timeouts: bool = False

    # Custom path to the Harbor tasks root (contains <instance_id>/tests/).
    # Defaults to <data_dir>/senior-swe-bench/tasks.
    tasks_dir: str | None = None


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_senior_swe_bench_generation_config", node=SeniorSweBenchGenerationConfig)


def parse_senior_swe_bench_reward(reward: dict | None, *, patch_exists: bool) -> dict:
    """Normalize Senior SWE-Bench Harbor reward artifacts into metrics fields.

    Empty ``reward.txt`` / missing structured reward after a verifier run is an
    invalid trial (infra or validation crash) and must not count as a solve.
    """
    if not patch_exists:
        return {
            "resolved": False,
            "patch_exists": False,
            "patch_successfully_applied": False,
            "reward": 0,
            "invalid_trial": False,
        }

    if not isinstance(reward, dict):
        return {
            "resolved": False,
            "patch_exists": True,
            "patch_successfully_applied": False,
            "reward": 0,
            "invalid_trial": True,
        }

    if reward.get("invalid_trial"):
        return {
            "resolved": False,
            "patch_exists": True,
            "patch_successfully_applied": True,
            "reward": 0,
            "invalid_trial": True,
            "raw_reward": reward,
        }

    apply_failed = bool(reward.get("apply_failed"))
    raw_reward = reward.get("reward", 0)
    try:
        reward_value = float(raw_reward) if raw_reward is not None else 0.0
    except (TypeError, ValueError):
        reward_value = 0.0

    metrics = {
        "resolved": (not apply_failed) and reward_value >= 1.0,
        "patch_exists": True,
        "patch_successfully_applied": not apply_failed,
        "reward": reward_value,
        "invalid_trial": False,
        "raw_reward": reward,
    }
    # Preserve optional flat SSB fields when present.
    for key in (
        "verifier_score",
        "rubric_score",
        "validation_score",
        "fail_to_pass_score",
        "pass_to_pass_score",
        "taste_score",
    ):
        if key in reward and reward[key] is not None:
            try:
                metrics[key] = float(reward[key])
            except (TypeError, ValueError):
                metrics[key] = reward[key]
    return metrics


def _verifier_env_apptainer_args(repo_name: str) -> str:
    """Build ``--env`` flags for API keys / overrides (Apptainer uses ``--cleanenv``)."""
    parts = [f"--env REPO_NAME={shlex.quote(repo_name)}"]
    for key in SSB_VERIFIER_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            parts.append(f"--env {key}={shlex.quote(value)}")
    return " " + " ".join(parts) + " "


class SeniorSweBenchGenerationTask(SweBenchGenerationTask):
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
        if mode == "agent" and self.cfg.use_agent_timeouts and data_point.get("agent_timeout_sec"):
            timeout = int(data_point.get("agent_timeout_sec")) + 120
        if mode == "eval" and self.cfg.use_verifier_timeouts and data_point.get("verifier_timeout_sec"):
            timeout = int(data_point.get("verifier_timeout_sec")) + 120
        return await SweBenchGenerationTask._execute_container_command(
            self, data_point, command, expected_file_pattern, mode, timeout, extra_apptainer_args
        )

    def _resolve_tests_dir(self, data_point: dict) -> Path:
        return resolve_tests_dir(
            data_point,
            self.cfg.eval_config,
            tasks_dir=self.cfg.tasks_dir,
            benchmark_name="senior-swe-bench",
        )

    async def _run_senior_swe_bench_verifier(self, data_point, model_patch: str) -> dict:
        """Apply model.patch under /repo/$REPO_NAME and run tests/test.sh."""
        patches_dir = self.output_dir / "patches"
        patches_dir.mkdir(parents=True, exist_ok=True)
        patch_path = patches_dir / f"{data_point['instance_id']}.patch"
        patch_path.write_text(model_patch if model_patch.endswith("\n") else model_patch + "\n")

        eval_out = self.output_dir / "eval-outputs" / data_point["instance_id"]
        if eval_out.exists():
            shutil.rmtree(eval_out)
        eval_out.mkdir(parents=True, exist_ok=True)

        repo_name = _validate_repo_name(
            data_point.get("repo_name") or Path(data_point.get("container_repo_dir", "/repo")).name
        )

        tests_dir = self._resolve_tests_dir(data_point)
        extra_apptainer_args = f" --mount type=bind,src={tests_dir},dst=/tests,ro "
        extra_apptainer_args += _verifier_env_apptainer_args(repo_name)

        instance_id = data_point["instance_id"]
        # SSB test.sh assumes a post-agent dirty tree under /repo/$REPO_NAME.
        # Apply the captured model.patch first; on failure write apply_failed reward.
        verifier_cmd = (
            "mkdir -p /logs/artifacts /logs/verifier && "
            f"cp /trajectories_mount/patches/{instance_id}.patch /logs/artifacts/model.patch && "
            "export TESTS_DIR=/tests && "
            "export VERIFIER_DIR=/logs/verifier && "
            "export ARTIFACTS_DIR=/logs/artifacts && "
            f"export REPO_NAME={repo_name} && "
            f"cd /repo/{repo_name} || {{ "
            '  echo \'{"reward": 0, "apply_failed": 1}\' > /logs/verifier/reward.json; '
            "  echo 0 > /logs/verifier/reward.txt; "
            f"  mkdir -p /trajectories_mount/eval-outputs/{instance_id} && "
            f"  cp -r /logs/verifier/. /trajectories_mount/eval-outputs/{instance_id}/; "
            "  exit 0; "
            "} && "
            "if ! git apply --whitespace=nowarn /logs/artifacts/model.patch 2>/logs/verifier/apply.err; then "
            "  if ! patch -p1 --forward --batch < /logs/artifacts/model.patch "
            "      >>/logs/verifier/apply.err 2>&1; then "
            '    echo \'{"reward": 0, "apply_failed": 1}\' > /logs/verifier/reward.json; '
            "    echo 0 > /logs/verifier/reward.txt; "
            f"    mkdir -p /trajectories_mount/eval-outputs/{instance_id} && "
            f"    cp -r /logs/verifier/. /trajectories_mount/eval-outputs/{instance_id}/; "
            "    exit 0; "
            "  fi; "
            "fi && "
            "bash /tests/test.sh; "
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
                extra_apptainer_args=extra_apptainer_args,
            )
        except ValueError:
            if not (eval_out / "reward.json").exists() and not (eval_out / "reward.txt").exists():
                LOG.error("Senior SWE-Bench verifier failed for %s", instance_id)
                return parse_senior_swe_bench_reward(None, patch_exists=True)

        reward = load_verifier_reward(eval_out)
        return parse_senior_swe_bench_reward(reward, patch_exists=True)

    async def process_single_datapoint(self, data_point, data, prompt_format=None):
        async with self.semaphore:
            pred_file = await self._run_agent(data_point)

        with open(pred_file, "r") as f:
            trajectory_dict = json.loads(f.read())

        model_patch = trajectory_dict["model_patch"]
        has_patch = bool(model_patch and str(model_patch).strip())

        if not has_patch:
            metrics = parse_senior_swe_bench_reward(None, patch_exists=False)
        elif not self.cfg.evaluate:
            metrics = {
                "resolved": None,
                "patch_exists": True,
                "patch_successfully_applied": None,
                "reward": None,
                "invalid_trial": None,
            }
        else:
            async with self.semaphore:
                metrics = await self._run_senior_swe_bench_verifier(data_point, str(model_patch))

        return {
            "swe-bench-metrics": metrics,
            "swe-bench-outputs": trajectory_dict,
            "generation": "",  # required TODO: we should fix this
        }


GENERATION_TASK_CLASS = SeniorSweBenchGenerationTask


@hydra.main(version_base=None, config_name="base_senior_swe_bench_generation_config")
def senior_swe_bench_generation(cfg: SeniorSweBenchGenerationConfig):
    cfg = SeniorSweBenchGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    SeniorSweBenchGenerationTask(cfg).generate()


HELP_MESSAGE = get_help_message(
    SeniorSweBenchGenerationConfig,
    server_params=server_params(),
)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        senior_swe_bench_generation()
