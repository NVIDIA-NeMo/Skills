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

"""Scale-SWE rollout generation with native, harness-free grading."""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
import sys
from pathlib import Path

import hydra

from nemo_skills.inference.eval import scale_swe_utils
from nemo_skills.inference.eval.scale_swe_utils import (
    build_scale_swe_metrics,
    normalize_scale_swe_data_point,
    normalize_scale_swe_report,
)
from nemo_skills.inference.eval.swebench import (
    SupportedAgentFrameworks,
    SweBenchGenerationConfig,
    SweBenchGenerationTask,
)
from nemo_skills.inference.model import server_params
from nemo_skills.prompt.utils import get_config_path
from nemo_skills.utils import get_help_message, get_logger_name, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))

cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_scale_swe_generation_config", node=SweBenchGenerationConfig)

_RUNNER = """\
from scale_swe_utils import run_scale_swe_evaluation
import sys

run_scale_swe_evaluation(sys.argv[1], sys.argv[2])
"""

SCALE_SWE_USER_PROMPT = """\
We are addressing the following issue in our repository. Please review the issue details below:

--- BEGIN ISSUE ---
{problem_statement}
--- END ISSUE ---

The repository is located at `{workspace_dir}`, and all your operations must be confined to this directory.
"""

_DEFAULT_AGENT_CONFIGS = {
    SupportedAgentFrameworks.swe_agent: "eval/scale-swe/swe-agent/default",
    SupportedAgentFrameworks.mini_swe_agent: "eval/scale-swe/mini-swe-agent/swebench",
}

_RESOLVER_CANDIDATES = (
    Path("/run/systemd/resolve/resolv.conf"),
    Path("/etc/resolv.conf"),
)


def format_scale_swe_user_prompt(problem_statement: str, workspace_dir: str = "/testbed") -> str:
    """Format the benchmark-level user message used by the official AweAgent recipe."""
    return SCALE_SWE_USER_PROMPT.format(
        problem_statement=problem_statement,
        workspace_dir=workspace_dir,
    )


class ScaleSweGenerationTask(SweBenchGenerationTask):
    """Reuse SWE-bench agents, then grade with Scale-SWE's F2P/P2P protocol."""

    def __init__(self, cfg: SweBenchGenerationConfig):
        if cfg.swe_zero_container is not None:
            raise ValueError(
                "Scale-SWE does not support swe_zero_container because native pre_commands and "
                "the per-instance parent_commit are required for valid rollouts."
            )
        if cfg.agent_config is None:
            cfg.agent_config = _DEFAULT_AGENT_CONFIGS.get(cfg.agent_framework)
        super().__init__(cfg)

    @staticmethod
    def _normalize_data_point(data_point: dict) -> dict:
        """Map native Scale-SWE aliases onto fields expected by the shared launcher."""
        return normalize_scale_swe_data_point(data_point)

    def _get_agent_problem_statement(self, data_point: dict) -> str:
        """Return the official Scale-SWE user message for direct-prompt harnesses."""
        return format_scale_swe_user_prompt(data_point.get("problem_statement", ""))

    def _get_openhands_instruction_template(self) -> str:
        """Return the Scale-SWE user template rendered by OpenHands."""
        return str(get_config_path("eval/scale-swe/openhands/swe_default", config_extension="j2"))

    def _get_apptainer_mounts(self, mode: str, data_point: dict) -> list[str]:
        """Use minimal mounts and a real resolver only for native Scale-SWE grading."""
        if mode != "eval":
            return super()._get_apptainer_mounts(mode, data_point)

        token = hashlib.sha256(str(data_point["instance_id"]).encode()).hexdigest()[:20]
        artifact_dir = self.output_dir / "scale-swe-eval" / token
        report_dir = self.output_dir / "eval-outputs" / token
        mounts = [
            f"type=bind,src={artifact_dir},dst=/scale_swe_eval,ro",
            f"type=bind,src={report_dir},dst=/scale_swe_report",
        ]
        if self.cfg.scale_swe_verifier_network:
            resolver = self._get_scale_swe_resolver()
            mounts.append(f"type=bind,src={resolver},dst=/etc/resolv.conf,ro")
        return mounts

    def _get_scale_swe_resolver(self) -> Path:
        """Select a resolver with upstream nameservers rather than a loopback-only stub."""
        configured = self.cfg.scale_swe_eval_resolv_conf
        candidates = (Path(configured),) if configured else _RESOLVER_CANDIDATES
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        candidate_list = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(f"No Scale-SWE evaluation resolver found. Checked: {candidate_list}")

    async def _run_scale_swe_verifier(self, data_point: dict, model_patch: str) -> dict:
        instance_id = str(data_point["instance_id"])
        token = hashlib.sha256(instance_id.encode()).hexdigest()[:20]
        artifact_dir = self.output_dir / "scale-swe-eval" / token
        eval_dir = self.output_dir / "eval-outputs" / token
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)
        if eval_dir.exists():
            shutil.rmtree(eval_dir)
        artifact_dir.mkdir(parents=True)
        eval_dir.mkdir(parents=True)

        model_patch_path = artifact_dir / "model.patch"
        model_patch_path.write_text(model_patch if model_patch.endswith("\n") else model_patch + "\n")

        config = {
            "workdir": data_point.get("workdir") or data_point.get("container_repo_dir", "/testbed"),
            "model_patch": "/scale_swe_eval/model.patch",
            "FAIL_TO_PASS": data_point.get("FAIL_TO_PASS"),
            "PASS_TO_PASS": data_point.get("PASS_TO_PASS"),
            "timeout": self.cfg.swebench_tests_timeout,
        }
        for field, filename in (("f2p_patch", "f2p.patch"), ("f2p_script", "f2p_script.py")):
            value = data_point.get(field)
            if value and str(value).strip():
                path = artifact_dir / filename
                path.write_text(str(value))
                config[field] = f"/scale_swe_eval/{filename}"

        config_path = artifact_dir / "config.json"
        config_path.write_text(json.dumps(config))
        (artifact_dir / "runner.py").write_text(_RUNNER)
        shutil.copyfile(Path(scale_swe_utils.__file__), artifact_dir / "scale_swe_utils.py")

        mounted_artifacts = "/scale_swe_eval"
        mounted_report = "/scale_swe_report/report.json"
        command = f"python {mounted_artifacts}/runner.py {mounted_artifacts}/config.json {mounted_report}"
        report_path = eval_dir / "report.json"
        try:
            await self._execute_container_command(
                data_point,
                command,
                str(report_path),
                mode="eval",
                timeout=self.cfg.swebench_tests_timeout + 120,
            )
        except ValueError:
            LOG.error("Scale-SWE verifier failed for %s", instance_id)
            return normalize_scale_swe_report(None)

        try:
            report = json.loads(report_path.read_text())
        except (OSError, json.JSONDecodeError):
            LOG.exception("Could not read Scale-SWE report for %s", instance_id)
            report = None
        return normalize_scale_swe_report(report)

    def _get_terminal_error_metrics(self, error: Exception) -> dict:
        """Represent an agent failure as a terminal unresolved Scale-SWE trial."""
        return build_scale_swe_metrics(
            patch_exists=False,
            patch_applied=False,
            resolved=False,
            details={
                "error": "generation_error",
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        )

    async def process_single_datapoint(self, data_point, data, prompt_format=None):
        data_point = self._normalize_data_point(data_point)
        async with self.semaphore:
            pred_file = await self._run_agent(data_point)

        try:
            trajectory_dict = json.loads(Path(pred_file).read_text())
        except (OSError, json.JSONDecodeError) as exc:
            LOG.error("Invalid Scale-SWE rollout for %s: %s", data_point.get("instance_id"), exc)
            trajectory_dict = {"model_patch": None, "error": str(exc)}

        model_patch = trajectory_dict.get("model_patch")
        has_patch = bool(model_patch and str(model_patch).strip())
        if not has_patch:
            metrics = build_scale_swe_metrics(
                patch_exists=False,
                patch_applied=False,
                resolved=False,
                details={"error": "empty_patch"},
            )
        elif not self.cfg.evaluate:
            metrics = build_scale_swe_metrics(
                patch_exists=True,
                patch_applied=None,
                resolved=None,
            )
        else:
            async with self.semaphore:
                metrics = await self._run_scale_swe_verifier(data_point, str(model_patch))

        return {
            "swe-bench-metrics": metrics,
            "swe-bench-outputs": trajectory_dict,
            "generation": "",
        }


GENERATION_TASK_CLASS = ScaleSweGenerationTask


@hydra.main(version_base=None, config_name="base_scale_swe_generation_config")
def scale_swe_generation(cfg: SweBenchGenerationConfig):
    cfg = SweBenchGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    ScaleSweGenerationTask(cfg).generate()


HELP_MESSAGE = get_help_message(
    SweBenchGenerationConfig,
    server_params=server_params(),
)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        scale_swe_generation()
