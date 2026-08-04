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

import asyncio
import glob
import json
import logging
import random
import shlex
import shutil
import sys
from pathlib import Path

import hydra

from nemo_skills.inference.eval.swebench import (
    SupportedAgentFrameworks,
    SweBenchGenerationConfig,
    SweBenchGenerationTask,
)
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class DeepSweGenerationConfig(SweBenchGenerationConfig):
    # DeepSWE does not use the SWE-bench eval harness.
    eval_harness_repo: str = ""
    eval_harness_commit: str = ""
    # Harbor verifier timeout fallback when a row omits verifier_timeout_sec.
    deepswe_tests_timeout: int = 60 * 30
    # eval_config.test_dir: optional root of Harbor task dirs (contains <instance_id>/tests/).
    # If unset, ns eval --data_dir resolves {data_dir}/deep-swe/tasks automatically.


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_deepswe_generation_config", node=DeepSweGenerationConfig)


def _resolve_container(data_point: dict) -> str:
    formatter = data_point["container_formatter"]
    return formatter.format(
        instance_id=data_point["instance_id"].replace("__", "_1776_"),
        task_id=data_point.get("task_id", data_point["instance_id"]),
        docker_image=data_point.get("docker_image", ""),
        docker_image_tag=str(data_point.get("docker_image", "")).rsplit(":", 1)[-1],
        ext_id=data_point.get("ext_id", ""),
    )


def _default_packaged_tasks_root() -> Path:
    # nemo_skills/inference/eval/deepswe.py -> nemo_skills/dataset/deep-swe/tasks
    return Path(__file__).resolve().parents[2] / "dataset" / "deep-swe" / "tasks"


def _get_eval_config_value(eval_config, key: str, default=None):
    if eval_config is None:
        return default
    if isinstance(eval_config, dict):
        return eval_config.get(key, default)
    return eval_config.get(key, default) if hasattr(eval_config, "get") else default


def _tests_candidate(tasks_root: Path, instance_id: str) -> Path:
    return Path(tasks_root) / instance_id / "tests"


def _resolve_tests_dir(data_point: dict, eval_config) -> Path:
    """Resolve Harbor tests/ for one task.

    Preference order:
    1. ``++eval_config.test_dir`` (explicit Harbor tasks root)
    2. ``++eval_config.data_dir``/deep-swe/tasks (set automatically by ``ns eval --data_dir``)
    3. Packaged / relative / absolute paths from prepare (local-only fallbacks)
    """
    instance_id = data_point["instance_id"]

    test_dir = _get_eval_config_value(eval_config, "test_dir")
    if test_dir:
        candidate = _tests_candidate(Path(test_dir), instance_id)
        if candidate.is_dir():
            return candidate
        raise ValueError(
            f"DeepSWE tests dir not found for {instance_id}: {candidate}. "
            f"Check ++eval_config.test_dir={test_dir} (expected <test_dir>/<instance_id>/tests)."
        )

    # ns eval --data_dir=... injects ++eval_config.data_dir; prefer that over stale
    # absolute paths baked into JSONL during prepare (those point at the prepare host).
    data_dir = _get_eval_config_value(eval_config, "data_dir")
    if data_dir:
        candidate = _tests_candidate(Path(data_dir) / "deep-swe" / "tasks", instance_id)
        if candidate.is_dir():
            return candidate
        raise ValueError(
            f"DeepSWE tests dir not found for {instance_id}: {candidate}. "
            f"Expected tasks under {{data_dir}}/deep-swe/tasks after "
            f"`ns prepare_data deep-swe --data_dir=...`. "
            f"Or pass ++eval_config.test_dir explicitly."
        )

    packaged = _tests_candidate(_default_packaged_tasks_root(), instance_id)
    if packaged.is_dir():
        return packaged

    rel = data_point.get("tests_dir_rel")
    if rel:
        candidate = Path(__file__).resolve().parents[2] / "dataset" / "deep-swe" / rel
        if candidate.is_dir():
            return candidate

    tests_dir = data_point.get("tests_dir")
    if tests_dir and Path(tests_dir).is_dir():
        return Path(tests_dir)

    raise ValueError(
        f"DeepSWE tests dir not found for {instance_id}. "
        "Pass --data_dir (recommended) or ++eval_config.test_dir=/path/to/deep-swe/tasks "
        "after ns prepare_data."
    )


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

    def __init__(self, cfg: DeepSweGenerationConfig):
        # Parent installs the SWE-bench harness when evaluate=True. DeepSWE grades via
        # Harbor tests/test.sh instead, so skip that install while still installing agents.
        evaluate = cfg.evaluate
        cfg.evaluate = False
        try:
            super().__init__(cfg)
        finally:
            cfg.evaluate = evaluate
            self.cfg.evaluate = evaluate

    async def _execute_container_command(self, data_point, command, expected_file_pattern, mode, timeout=100000):
        """Agent mode mirrors SWE-bench (/app -> /testbed); eval mode runs Harbor verifier."""
        if mode == "agent":
            return await self._execute_agent_container_command(
                data_point, command, expected_file_pattern, timeout=timeout
            )
        if mode == "eval":
            return await self._execute_harbor_eval_command(data_point, command, expected_file_pattern, timeout=timeout)
        raise ValueError(f"Unsupported DeepSWE container mode: {mode}")

    async def _execute_agent_container_command(self, data_point, command, expected_file_pattern, timeout=100000):
        """Same agent Apptainer setup as SWE-bench, with DeepSWE container placeholders."""
        # Resolve DeepSWE formatter placeholders, then reuse SWE-bench agent mounts/copy-to-/testbed.
        resolved = {**data_point, "container_formatter": _resolve_container(data_point)}
        if timeout >= 100000:
            timeout = int(data_point.get("agent_timeout_sec") or 5400) + 120
        return await SweBenchGenerationTask._execute_container_command(
            self, resolved, command, expected_file_pattern, mode="agent", timeout=timeout
        )

    async def _execute_harbor_eval_command(self, data_point, command, expected_file_pattern, timeout=100000):
        """Fresh task image + Harbor tests/ bind-mount; does not reuse the agent /testbed copy."""
        container_commands = ["echo '127.0.0.1 localhost' >/etc/hosts"]
        tests_dir = _resolve_tests_dir(data_point, self.cfg.eval_config)
        if not tests_dir.is_dir():
            raise ValueError(
                f"DeepSWE tests dir not found for {data_point['instance_id']}: {tests_dir}. "
                "Pass --data_dir or ++eval_config.test_dir=<data_dir>/deep-swe/tasks"
            )

        extra_apptainer_args = f" --mount type=bind,src={tests_dir},dst=/tests,ro "
        container_commands.append("mkdir -p /logs/artifacts /logs/verifier")
        container_commands.append(
            f"cp /trajectories_mount/patches/{data_point['instance_id']}.patch /logs/artifacts/model.patch"
        )
        container_commands.append(command)
        combined_command = " && ".join(container_commands)
        container_name = _resolve_container(data_point)

        apptainer_cmd = (
            f"apptainer exec --writable-tmpfs --cleanenv --no-mount home,tmp,bind-paths "
            f"--mount type=bind,src=/nemo_run/code,dst=/nemo_run/code "
            f"--mount type=bind,src={Path(self.cfg.input_file).parent},dst=/input_mount,ro "
            f"--mount type=bind,src=/root,dst=/root_mount,ro "
            f"--mount type=bind,src={self.output_dir},dst=/trajectories_mount "
            f"{extra_apptainer_args} "
            f"{container_name} bash -c {shlex.quote(combined_command)}"
        )

        logs_dir = self.output_dir / "apptainer_logs"
        logs_dir.mkdir(exist_ok=True)

        for attempt in range(self.cfg.max_retries):
            log_file_path = logs_dir / f"{data_point['instance_id']}_eval_attempt{attempt + 1}.log"
            LOG.info(
                "Starting DeepSWE Harbor verifier (attempt %d of %d). Logs: %s",
                attempt + 1,
                self.cfg.max_retries,
                log_file_path,
            )
            pred_files: list[str] = []
            try:
                with open(log_file_path, "w") as log_file:
                    process = await asyncio.create_subprocess_shell(apptainer_cmd, stdout=log_file, stderr=log_file)
                    try:
                        await asyncio.wait_for(process.communicate(), timeout=timeout)
                    except asyncio.TimeoutError:
                        if process.returncode is None:
                            process.kill()
                            await process.wait()
                        raise ValueError("Command timed out")

                pred_files = glob.glob(expected_file_pattern, recursive=True)
                # DeepSWE may write reward.json and/or crash-sentinel reward.txt.
                # Prefer reward.json when both exist.
                if pred_files:
                    preferred = [p for p in pred_files if p.endswith("reward.json")]
                    return preferred[0] if preferred else pred_files[0]
                raise ValueError(
                    f"Expected a file matching {expected_file_pattern} for "
                    f"{data_point['instance_id']}, found {len(pred_files)}."
                )
            except Exception:
                if attempt < self.cfg.max_retries - 1:
                    retry_interval = random.randint(self.cfg.min_retry_interval, self.cfg.max_retry_interval)
                    LOG.warning(
                        "Attempt %d failed for DeepSWE instance %s. Retrying in %d seconds...",
                        attempt + 1,
                        data_point["instance_id"],
                        retry_interval,
                    )
                    if retry_interval > 0:
                        await asyncio.sleep(retry_interval)
                    continue
                LOG.error("All %d attempts failed for instance %s", self.cfg.max_retries, data_point["instance_id"])
                LOG.error("Apptainer command failed. Check logs at: %s", log_file_path)
                raise ValueError(
                    f"DeepSWE verifier failed for {data_point['instance_id']}. Check logs at: {log_file_path}. "
                    f"Expected a file matching {expected_file_pattern}, "
                    f"found {len(pred_files)}."
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

        verifier_cmd = (
            "export TESTS_DIR=/tests && "
            "export VERIFIER_DIR=/logs/verifier && "
            "export APP_DIR=/app && "
            "export ARTIFACTS_DIR=/logs/artifacts && "
            "cd /app && "
            "bash /tests/test.sh; "
            f"mkdir -p /trajectories_mount/eval-outputs/{data_point['instance_id']} && "
            f"cp -r /logs/verifier/. /trajectories_mount/eval-outputs/{data_point['instance_id']}/"
        )

        timeout = int(data_point.get("verifier_timeout_sec") or self.cfg.deepswe_tests_timeout)
        # Prefer reward.json; DeepSWE crash sentinel only writes reward.txt=-1.
        search_path = str(eval_out / "reward.*")
        try:
            await self._execute_container_command(
                data_point,
                verifier_cmd,
                search_path,
                mode="eval",
                timeout=timeout + 120,
            )
        except ValueError:
            if not (eval_out / "reward.json").exists() and not (eval_out / "reward.txt").exists():
                LOG.error("DeepSWE verifier failed for %s", data_point["instance_id"])
                return parse_deepswe_reward(None, patch_exists=True)

        reward = _load_verifier_reward(eval_out)
        return parse_deepswe_reward(reward, patch_exists=True)

    async def _run_agent(self, data_point, api_base) -> str:
        """Dispatch to the same agent runners as SWE-bench."""
        if self.cfg.agent_framework == SupportedAgentFrameworks.swe_agent:
            return await self._run_swe_agent(data_point, api_base)
        if self.cfg.agent_framework == SupportedAgentFrameworks.mini_swe_agent:
            return await self._run_mini_swe_agent(data_point, api_base)
        if self.cfg.agent_framework == SupportedAgentFrameworks.openhands:
            return await self._run_openhands(data_point, api_base)
        if self.cfg.agent_framework == SupportedAgentFrameworks.gold_patch:
            return await self._get_gold_patch(data_point)
        raise ValueError(
            f"Unsupported agent framework: {self.cfg.agent_framework}. "
            f"Supported: {', '.join(f.value for f in SupportedAgentFrameworks)}."
        )

    async def process_single_datapoint(self, data_point, data, prompt_format=None):
        if "base_url" in self.cfg.server:
            api_base = self.cfg.server.base_url
        else:
            api_base = f"http://{self.cfg.server.host}:{self.cfg.server.port}/v1"

        async with self.semaphore:
            pred_file = await self._run_agent(data_point, api_base)

        with open(pred_file, "r") as f:
            trajectory_dict = json.loads(f.read())

        model_patch = trajectory_dict.get("model_patch")
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
