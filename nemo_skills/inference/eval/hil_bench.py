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

"""HiL-Bench SWE: hil-bench SWE-agent generation + Harbor verifier grading.

Generation uses the SWE-agent checkout vendored in
https://github.com/hilbenchauthors/hil-bench (``SWE-agent/`` subdirectory).
Only the ``swe_agent`` harness is supported.

Grading is decoupled from the SWE-bench harness: each task is scored with its
Harbor ``tests/test.sh`` verifier (same pattern as DeepSWE).
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import sys
from pathlib import Path

import hydra

from nemo_skills.inference.eval.swebench import (
    SupportedAgentFrameworks,
    SupportedDatasetTypes,
    SweBenchGenerationConfig,
    SweBenchGenerationTask,
)
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))

DEFAULT_HIL_BENCH_REPO = "https://github.com/hilbenchauthors/hil-bench.git"


@nested_dataclass(kw_only=True)
class HilBenchGenerationConfig(SweBenchGenerationConfig):
    # HiL-Bench only supports the hil-bench SWE-agent harness.
    agent_framework: SupportedAgentFrameworks = SupportedAgentFrameworks.swe_agent
    agent_framework_repo: str | None = DEFAULT_HIL_BENCH_REPO
    agent_framework_commit: str | None = "HEAD"
    # Path to the SWE-agent package inside the hil-bench repository.
    agent_framework_subdir: str = "SWE-agent"

    dataset_type: SupportedDatasetTypes = SupportedDatasetTypes.hil_bench

    # Whether to apply timeouts on the agent and verifier if set on the data point.
    use_agent_timeouts: bool = False
    use_verifier_timeouts: bool = False

    # Custom path to the Harbor tasks root (contains <instance_id>/tests/).
    # Defaults to <data_dir>/hil-swe-bench/tasks.
    tasks_dir: str | None = None

    # Optional ask_human judge server URL (host-reachable from the task container).
    # When unset, ask_human mode still runs but the tool will fail if called.
    ask_human_server_url: str | None = None


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_hil_bench_generation_config", node=HilBenchGenerationConfig)


def parse_hil_reward(reward: dict | None, *, patch_exists: bool) -> dict:
    """Normalize Harbor reward.json / reward.txt into NeMo-Skills metrics fields."""
    if not patch_exists:
        return {
            "resolved": False,
            "patch_exists": False,
            "patch_successfully_applied": False,
            "reward": 0,
            "fail_to_pass_passed": 0,
            "fail_to_pass_total": 0,
        }

    if not isinstance(reward, dict):
        return {
            "resolved": False,
            "patch_exists": True,
            "patch_successfully_applied": False,
            "reward": 0,
            "fail_to_pass_passed": 0,
            "fail_to_pass_total": 0,
        }

    apply_failed = bool(reward.get("apply_failed"))
    raw_reward = reward.get("reward", reward.get("resolved", 0))
    try:
        reward_value = float(raw_reward)
    except (TypeError, ValueError):
        reward_value = 0.0

    # Crash / infra sentinel (negative reward), matching DeepSWE convention.
    if reward_value < 0:
        return {
            "resolved": False,
            "patch_exists": True,
            "patch_successfully_applied": False,
            "reward": reward_value,
            "fail_to_pass_passed": 0,
            "fail_to_pass_total": 0,
            "verifier_crashed": True,
            "raw_reward": reward,
        }

    resolved_raw = reward.get("resolved", reward_value)
    try:
        resolved_value = float(resolved_raw)
    except (TypeError, ValueError):
        resolved_value = reward_value

    metrics = {
        "resolved": (not apply_failed) and resolved_value >= 1.0,
        "patch_exists": True,
        "patch_successfully_applied": not apply_failed,
        "reward": reward_value,
        "fail_to_pass_passed": int(reward.get("fail_to_pass_passed") or 0),
        "fail_to_pass_total": int(reward.get("fail_to_pass_total") or 0),
        "raw_reward": reward,
    }

    # Optional ask_human sidecar metrics merged by Harbor test.sh.
    for key in ("n_questions", "n_blockers", "blockers_resolved", "precision", "recall", "f1", "ask_f1"):
        if key in reward:
            metrics[key] = reward[key]
    if "f1" in metrics and "ask_f1" not in metrics:
        metrics["ask_f1"] = metrics["f1"]

    return metrics


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
            return {"reward": float(raw), "resolved": float(raw)}
        except ValueError:
            return {"reward": 0, "resolved": 0}
    return None


class HilBenchGenerationTask(SweBenchGenerationTask):
    """hil-bench SWE-agent for generation; Harbor ``tests/test.sh`` for grading."""

    def __init__(self, cfg: HilBenchGenerationConfig):
        if cfg.agent_framework != SupportedAgentFrameworks.swe_agent:
            raise ValueError(f"hil-swe-bench only supports ++agent_framework=swe_agent (got {cfg.agent_framework!r}).")
        if cfg.agent_framework_repo is None:
            cfg.agent_framework_repo = DEFAULT_HIL_BENCH_REPO
        if cfg.agent_framework_commit is None:
            cfg.agent_framework_commit = "HEAD"
        if cfg.dataset_type != SupportedDatasetTypes.hil_bench:
            cfg.dataset_type = SupportedDatasetTypes.hil_bench

        # Force swe_agent defaults so SweBenchGenerationTask installs SWE-agent,
        # then rewrite the clone/install command to use hil-bench's SWE-agent subdir.
        # We do this by temporarily monkeypatching the setup list construction via
        # a pre-super hook on the clone repo path: clone hil-bench into a temp dir
        # name, then let the normal install run against the extracted SWE-agent.
        #
        # Concrete approach: set agent_framework_repo to hil-bench and override the
        # install command by replacing SweBenchGenerationTask.__init__ setup for
        # swe_agent after calling a trimmed init path.
        self._hil_cfg = cfg
        self._install_hil_swe_agent_then_init(cfg)

    def _install_hil_swe_agent_then_init(self, cfg: HilBenchGenerationConfig):
        """Run SweBenchGenerationTask.__init__ with hil-bench SWE-agent install."""
        # Patch the default SWE-agent install to extract the vendored subdirectory.
        # SweBenchGenerationTask clones agent_framework_repo into /root/SWE-agent.
        # For hil-bench we clone the monorepo then move SWE-agent into place.
        original_repo = cfg.agent_framework_repo
        original_commit = cfg.agent_framework_commit

        # Use a sentinel: after super().__init__ would clone hil-bench into /root/SWE-agent
        # (wrong layout). Instead, override by running custom setup before GenerationTask
        # pieces. Easiest reliable approach: call GenerationTask-compatible init by
        # temporarily replacing the swe_agent setup block via cfg trick:
        # clone into /root/hil-bench-src then rearrange before uv install.
        #
        # We achieve this by setting agent_framework_repo and replacing __init__ body
        # for setup_commands only — copy the relevant parts from parent.

        # Call parent __init__ but intercept: set repo to hil-bench and commit, then
        # after parent runs, if /root/SWE-agent/SWE-agent exists, rearrange.
        # Parent installs with `uv pip install -e .` at /root/SWE-agent which fails
        # if it's the hil-bench root. So we must customize BEFORE parent install.
        #
        # Solution: set agent_framework to gold_patch briefly to skip agent install,
        # call parent for evaluate=False harness skip + common setup, then install
        # hil SWE-agent ourselves. But parent also sets up uv and output dirs.
        evaluate = cfg.evaluate
        cfg.evaluate = False  # skip SWE-bench harness
        saved_framework = cfg.agent_framework
        cfg.agent_framework = SupportedAgentFrameworks.gold_patch
        SweBenchGenerationTask.__init__(self, cfg)
        cfg.agent_framework = saved_framework
        cfg.evaluate = evaluate
        self.cfg = cfg

        subdir = cfg.agent_framework_subdir.strip("/") or "SWE-agent"
        setup_commands = [
            "curl -Lf https://astral.sh/uv/install.sh | sh && "
            "source /root/.local/bin/env && "
            "export UV_PYTHON_INSTALL_DIR=/root/uv/python && "
            "export UV_TOOL_DIR=/root/uv/tool && "
            "export UV_TOOL_BIN_DIR=/root/uv/tool-bin",
            "rm -rf /root/hil-bench-src /root/SWE-agent && "
            f"git clone {original_repo} /root/hil-bench-src && "
            "cd /root/hil-bench-src && "
            f"git checkout {original_commit} && "
            f"test -d /root/hil-bench-src/{subdir} && "
            f"mv /root/hil-bench-src/{subdir} /root/SWE-agent && "
            "rm -rf /root/hil-bench-src && "
            "cd /root/SWE-agent && "
            "uv venv --python 3.12 --managed-python venv && "
            "source venv/bin/activate && "
            "uv pip install -e . && "
            "uv pip install rich==14.2.0",
        ]
        asyncio.run(self._execute_local_command(" && ".join(setup_commands), timeout=self.cfg.setup_timeout))

        if "base_url" in self.cfg.server:
            self.api_base = self.cfg.server.base_url
        else:
            self.api_base = f"http://{self.cfg.server.host}:{self.cfg.server.port}/v1"

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
        )

    def _default_agent_config_for_mode(self, mode: str) -> str:
        if mode == "ask_human":
            return "eval/hil-swe-bench/swe-agent/ask_human"
        # baseline and full_info share the same agent YAML; full_info differences
        # are already baked into problem_statement by prepare.py.
        return "eval/hil-swe-bench/swe-agent/default"

    def _resolve_tests_dir(self, data_point: dict) -> Path:
        """Resolve Harbor tests/ for one HiL task."""
        instance_id = data_point["instance_id"]

        if self.cfg.tasks_dir:
            candidate = Path(self.cfg.tasks_dir) / instance_id / "tests"
            if candidate.is_dir():
                return candidate
            raise ValueError(
                f"HiL-Bench tests dir not found for {instance_id}: {candidate}. "
                f"Check ++tasks_dir={self.cfg.tasks_dir}."
            )

        data_dir = self.cfg.eval_config.get("data_dir")
        if data_dir:
            candidate = Path(data_dir) / "hil-swe-bench" / "tasks" / instance_id / "tests"
            if candidate.is_dir():
                return candidate
            raise ValueError(
                f"HiL-Bench tests dir not found for {instance_id}: {candidate}. "
                f"Expected tasks under {{data_dir}}/hil-swe-bench/tasks after "
                f"`ns prepare_data hil-swe-bench --data_dir=...`. "
                f"Or pass ++tasks_dir explicitly."
            )

        raise ValueError(
            f"HiL-Bench tests dir not found for {instance_id}. "
            "Pass --data_dir (recommended) or ++tasks_dir=/path/to/hil-swe-bench/tasks "
            "after ns prepare_data."
        )

    async def _run_swe_agent(self, data_point, api_base):
        saved_config = self.cfg.agent_config
        try:
            if self.cfg.agent_config is None:
                mode = data_point.get("mode") or "baseline"
                self.cfg.agent_config = self._default_agent_config_for_mode(mode)

            # ask_human tool expects these env vars inside the agent process.
            mode = data_point.get("mode") or "baseline"
            env_exports = []
            if mode == "ask_human":
                env_exports.append(f"export TASK_INSTANCE_ID={data_point['instance_id']}")
                server_url = self.cfg.ask_human_server_url
                if server_url:
                    env_exports.append(f"export ASK_HUMAN_SERVER_URL={server_url}")

            if env_exports:
                # Wrap parent command by temporarily patching via prepending exports in
                # a thin override of the command construction: call parent then...
                # Parent builds the full command internally. Inject via pre_commands on
                # a shallow copy of the data point.
                data_point = dict(data_point)
                pre = data_point.get("pre_commands", "").strip()
                inject = " && ".join(env_exports)
                data_point["pre_commands"] = f"{pre} && {inject}" if pre else inject

            return await SweBenchGenerationTask._run_swe_agent(self, data_point, api_base)
        finally:
            self.cfg.agent_config = saved_config

    async def _run_hil_verifier(self, data_point, model_patch: str) -> dict:
        """Apply model.patch in a pristine task image and run tests/test.sh."""
        patches_dir = self.output_dir / "patches"
        patches_dir.mkdir(parents=True, exist_ok=True)
        patch_path = patches_dir / f"{data_point['instance_id']}.patch"
        patch_path.write_text(model_patch if model_patch.endswith("\n") else model_patch + "\n")

        eval_out = self.output_dir / "eval-outputs" / data_point["instance_id"]
        if eval_out.exists():
            shutil.rmtree(eval_out)
        eval_out.mkdir(parents=True, exist_ok=True)

        tests_dir = self._resolve_tests_dir(data_point)
        repo_dir = data_point.get("container_repo_dir", "/app")
        extra_apptainer_args = f" --mount type=bind,src={tests_dir},dst=/tests,ro "

        # HiL Harbor test.sh applies the hidden test_patch and runs FAIL_TO_PASS tests,
        # but does not apply the agent patch — do that explicitly first.
        verifier_cmd = (
            "mkdir -p /logs/artifacts /logs/verifier && "
            f"cp /trajectories_mount/patches/{data_point['instance_id']}.patch "
            f"/logs/artifacts/model.patch && "
            f"cd {repo_dir} && "
            "git config --global --add safe.directory "
            f"{repo_dir} && "
            "git apply --verbose /logs/artifacts/model.patch || {"
            "  echo '[verifier] model.patch failed to apply' >&2;"
            "  echo 0 > /logs/verifier/reward.txt;"
            '  echo \'{"resolved":0,"reward":0,"apply_failed":1}\' > /logs/verifier/reward.json;'
            f"  mkdir -p /trajectories_mount/eval-outputs/{data_point['instance_id']};"
            f"  cp -r /logs/verifier/. /trajectories_mount/eval-outputs/{data_point['instance_id']}/;"
            "  exit 0;"
            "} && "
            "export TESTS_DIR=/tests && "
            "export VERIFIER_DIR=/logs/verifier && "
            f"export APP_DIR={repo_dir} && "
            "export ARTIFACTS_DIR=/logs/artifacts && "
            "bash /tests/test.sh; "
            f"mkdir -p /trajectories_mount/eval-outputs/{data_point['instance_id']} && "
            f"cp -r /logs/verifier/. /trajectories_mount/eval-outputs/{data_point['instance_id']}/"
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
                LOG.error("HiL-Bench verifier failed for %s", data_point["instance_id"])
                return parse_hil_reward(None, patch_exists=True)

        reward = _load_verifier_reward(eval_out)
        return parse_hil_reward(reward, patch_exists=True)

    async def process_single_datapoint(self, data_point, data, prompt_format=None):
        if "base_url" in self.cfg.server:
            api_base = self.cfg.server.base_url
        else:
            api_base = f"http://{self.cfg.server.host}:{self.cfg.server.port}/v1"

        async with self.semaphore:
            pred_file = await self._run_swe_agent(data_point, api_base)

        with open(pred_file, "r") as f:
            trajectory_dict = json.loads(f.read())

        model_patch = trajectory_dict.get("model_patch")
        has_patch = bool(model_patch and str(model_patch).strip())

        if not has_patch:
            metrics = parse_hil_reward(None, patch_exists=False)
        elif not self.cfg.evaluate:
            metrics = {
                "resolved": None,
                "patch_exists": True,
                "patch_successfully_applied": None,
                "reward": None,
                "fail_to_pass_passed": None,
                "fail_to_pass_total": None,
            }
        else:
            metrics = await self._run_hil_verifier(data_point, str(model_patch))

        return {
            "swe-bench-metrics": metrics,
            "swe-bench-outputs": trajectory_dict,
            "generation": "",
        }


GENERATION_TASK_CLASS = HilBenchGenerationTask


@hydra.main(version_base=None, config_name="base_hil_bench_generation_config")
def hil_bench_generation(cfg: HilBenchGenerationConfig):
    cfg = HilBenchGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    HilBenchGenerationTask(cfg).generate()


HELP_MESSAGE = get_help_message(
    HilBenchGenerationConfig,
    server_params=server_params(),
)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        hil_bench_generation()
