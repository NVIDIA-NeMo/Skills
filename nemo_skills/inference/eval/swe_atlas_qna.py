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

import asyncio
import glob
import json
import logging
import random
import shlex
import shutil
import sys
import tempfile
from dataclasses import field
from pathlib import Path

import hydra
from omegaconf import OmegaConf

from nemo_skills.inference.eval.swebench import (
    SupportedAgentFrameworks,
    SweBenchGenerationConfig,
    SweBenchGenerationTask,
    SweBenchInferenceConfig,
)
from nemo_skills.inference.model import server_params
from nemo_skills.prompt.utils import get_config_path
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))

FINAL_ANSWER_TAG = "<<FINAL_ANSWER>>"
# Directories each harness installs under /root; OpenSandbox stages them at /root_mount.
OPENSANDBOX_RUNTIME_DIRS = {
    SupportedAgentFrameworks.mini_swe_agent: ("mini-swe-agent", "uv"),
    SupportedAgentFrameworks.swe_agent: ("SWE-agent", "uv"),
    SupportedAgentFrameworks.openhands: ("OpenHands", "uv", "tmux", "jq"),
    SupportedAgentFrameworks.opencode: ("opencode", "node"),
    SupportedAgentFrameworks.claude_code: ("claude-code", "node"),
}
# Helper scripts that agent setup and commands run from /nemo_run/code inside task containers.
OPENSANDBOX_CODE_UPLOADS = ("openhands_alpine.py", "openhands_compat.py", "openhands_atlas.py")
NEMO_RUN_EVAL_DIR = "/nemo_run/code/nemo_skills/inference/eval"
DEFAULT_AGENT_CONFIGS = {
    SupportedAgentFrameworks.mini_swe_agent: "eval/swe-atlas-qna/mini-swe-agent/default",
    SupportedAgentFrameworks.swe_agent: "eval/swe-atlas-qna/swe-agent/default",
    SupportedAgentFrameworks.openhands: "eval/swe-atlas-qna/openhands/default",
    SupportedAgentFrameworks.opencode: "eval/swe-atlas-qna/opencode/default",
    SupportedAgentFrameworks.claude_code: "eval/swe-atlas-qna/claude-code/default",
}


def extract_final_answer(submission: str | None) -> str:
    """Extract the answer between the first and last FINAL_ANSWER tags."""
    if not submission:
        return ""

    first_tag = submission.find(FINAL_ANSWER_TAG)
    last_tag = submission.rfind(FINAL_ANSWER_TAG)
    if first_tag == -1:
        return submission.strip()
    if first_tag == last_tag:
        LOG.warning("Submitted answer contains only one %s tag; keeping the full submission", FINAL_ANSWER_TAG)
        return submission.strip()

    return submission[first_tag + len(FINAL_ANSWER_TAG) : last_tag].strip()


def extract_openhands_final_response(history: list[dict] | None) -> str:
    """Extract the last useful agent response from an OpenHands 1.2.1 history."""
    candidates = []
    for event in reversed(history or []):
        if event.get("source") != "agent":
            continue
        args = event.get("args") if isinstance(event.get("args"), dict) else {}
        if event.get("action") == "finish":
            if isinstance(args.get("final_thought"), str) and args["final_thought"].strip():
                candidates.append(args["final_thought"])
            outputs = args.get("outputs")
            if isinstance(outputs, dict):
                for key in ("content", "final_response", "answer"):
                    if isinstance(outputs.get(key), str) and outputs[key].strip():
                        candidates.append(outputs[key])
                candidates.extend(value for value in outputs.values() if isinstance(value, str) and value.strip())
            if isinstance(args.get("thought"), str) and args["thought"].strip():
                candidates.append(args["thought"])
        elif event.get("action") == "message":
            content = args.get("content", event.get("message"))
            if isinstance(content, str) and content.strip():
                candidates.append(content)

    for candidate in candidates:
        if FINAL_ANSWER_TAG in candidate:
            return candidate
    return candidates[0] if candidates else ""


@nested_dataclass(kw_only=True)
class SweAtlasQnAGenerationConfig(SweBenchGenerationConfig):
    inference: SweBenchInferenceConfig = field(default_factory=SweBenchInferenceConfig)
    # Where agent rollouts run: local Apptainer task containers or remote OpenSandbox task containers.
    execution_backend: str = "opensandbox"  # opensandbox | apptainer
    # OpenSandboxExecutor options, e.g. resources.cpu or block_network; credentials come from the environment.
    opensandbox: dict = field(default_factory=dict)
    # Coordinator interface that sandboxes use to reach the LLM proxy. Auto-detected inside the job if None.
    opensandbox_proxy_host: str | None = None


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_swe_atlas_qna_generation_config", node=SweAtlasQnAGenerationConfig)


class SweAtlasQnAGenerationTask(SweBenchGenerationTask):
    """Run a repository agent and return its submitted prose answer."""

    def __init__(self, cfg: SweAtlasQnAGenerationConfig):
        execution_backend = cfg.execution_backend
        if execution_backend not in {"apptainer", "opensandbox"}:
            raise ValueError(f"Unknown execution_backend: {execution_backend}")
        self.opensandbox_executor = None
        if execution_backend == "opensandbox":
            from nemo_skills.inference.eval.opensandbox import OpenSandboxExecutor

            if cfg.swe_zero_container is not None:
                raise ValueError("SWE-Atlas-QnA does not support swe_zero_container with OpenSandbox.")
            options = cfg.opensandbox
            if not isinstance(options, dict):
                options = OmegaConf.to_container(options, resolve=True)
            self.opensandbox_executor = OpenSandboxExecutor(**options)
            if cfg.opensandbox_proxy_host is None:
                cfg.opensandbox_proxy_host = self.opensandbox_executor.detect_proxy_host()
                LOG.info("Detected OpenSandbox model proxy interface: %s", cfg.opensandbox_proxy_host)
            if cfg.opensandbox_proxy_host in {"", "localhost", "127.0.0.1", "0.0.0.0", "::1"}:
                raise ValueError("Set opensandbox_proxy_host to a coordinator interface reachable from OpenSandbox.")
        if cfg.agent_framework not in DEFAULT_AGENT_CONFIGS:
            supported_frameworks = ", ".join(framework.value for framework in DEFAULT_AGENT_CONFIGS)
            raise ValueError(f"SWE-Atlas-QnA supports only these agent frameworks: {supported_frameworks}")
        if cfg.agent_config is None:
            cfg.agent_config = DEFAULT_AGENT_CONFIGS[cfg.agent_framework]
        if cfg.evaluate:
            LOG.warning(
                "SWE-Atlas-QnA does not support inline evaluation; overriding evaluate=True with evaluate=False"
            )
        cfg.evaluate = False
        super().__init__(cfg)

    def _run_setup(self, setup_commands: list[str]) -> None:
        if self.opensandbox_executor is None:
            super()._run_setup(setup_commands)
            return
        # Each sandbox is a fresh task image, so the installers run there before every rollout.
        self.opensandbox_setup_command = " && ".join(setup_commands)

    def _get_llm_proxy_host(self) -> str:
        if self.opensandbox_executor is None:
            return super()._get_llm_proxy_host()
        return self.cfg.opensandbox_proxy_host

    def _get_opensandbox_uploads(self) -> list[tuple[Path, str]]:
        """Files that agent commands read from coordinator paths, keyed by their path inside the sandbox."""
        eval_dir = Path(__file__).parent
        uploads = [(eval_dir / name, f"{NEMO_RUN_EVAL_DIR}/{name}") for name in OPENSANDBOX_CODE_UPLOADS]
        if self.cfg.agent_framework == SupportedAgentFrameworks.swe_agent:
            config_path = get_config_path(self.cfg.agent_config)
            uploads.append((Path(config_path), str(config_path)))
        return uploads

    def _get_opensandbox_repo_setup(self, data_point: dict) -> str:
        """Mirror the Apptainer repository preparation: pre_commands, then expose the repo at /testbed."""
        container_repo_dir = data_point.get("container_repo_dir", "/testbed")
        commands = []
        pre_commands = data_point.get("pre_commands", "").strip()
        if pre_commands:
            commands.extend([f"cd {shlex.quote(container_repo_dir)}", pre_commands])
        if container_repo_dir != "/testbed":
            commands.append(f"cp -r {shlex.quote(container_repo_dir)} /testbed")
        return " && ".join(commands)

    async def _execute_container_command(self, data_point, command, expected_file_pattern, mode, timeout=100000):
        if self.opensandbox_executor is None:
            return await super()._execute_container_command(
                data_point, command, expected_file_pattern, mode, timeout=timeout
            )
        if mode != "agent":
            raise ValueError("SWE-Atlas-QnA runs only agent rollouts in OpenSandbox.")

        from nemo_skills.inference.eval.opensandbox import OpenSandboxCommandTimeout, get_task_image

        image = get_task_image(data_point)
        logs_dir = self.output_dir / "opensandbox_logs"
        for attempt in range(self.cfg.max_retries):
            log_file = logs_dir / f"{data_point['instance_id']}_{mode}_attempt{attempt + 1}.log"
            LOG.info(
                "Starting OpenSandbox rollout (attempt %d of %d). Logs are available at %s",
                attempt + 1,
                self.cfg.max_retries,
                log_file,
            )
            staged_output = Path(tempfile.mkdtemp(dir=self.scratch_dir))
            try:
                await self.opensandbox_executor.run_agent(
                    instance_id=data_point["instance_id"],
                    image=image,
                    setup_command=self.opensandbox_setup_command,
                    runtime_dirs=OPENSANDBOX_RUNTIME_DIRS[self.cfg.agent_framework],
                    uploads=self._get_opensandbox_uploads(),
                    repo_setup_command=self._get_opensandbox_repo_setup(data_point),
                    command=command,
                    timeout=timeout,
                    setup_timeout=self.cfg.setup_timeout,
                    proxy_host=self.cfg.opensandbox_proxy_host,
                    outputs_destination=staged_output,
                    log_file=log_file,
                )
                await asyncio.to_thread(shutil.copytree, staged_output, self.output_dir, dirs_exist_ok=True)
                pred_files = glob.glob(expected_file_pattern, recursive=True)
                if len(pred_files) == 1:
                    return pred_files[0]
                raise ValueError(
                    f"Expected exactly one file matching {expected_file_pattern} for {data_point['instance_id']}, "
                    f"found {len(pred_files)}."
                )
            except OpenSandboxCommandTimeout as error:
                raise ValueError("Command timed out") from error
            except Exception as error:
                if attempt >= self.cfg.max_retries - 1:
                    raise ValueError(
                        f"Job failed for {data_point['instance_id']}. Check logs at: {log_file}."
                    ) from error
                retry_interval = random.randint(self.cfg.min_retry_interval, self.cfg.max_retry_interval)
                LOG.warning(
                    "Attempt %d failed for instance %s (%s). Retrying in %d seconds...",
                    attempt + 1,
                    data_point["instance_id"],
                    error,
                    retry_interval,
                )
                if retry_interval > 0:
                    await asyncio.sleep(retry_interval)
            finally:
                shutil.rmtree(staged_output, ignore_errors=True)

    def get_error_output(self, error: Exception, data_point: dict) -> dict:
        """Persist terminal rollout failures so resume and scoring retain the task."""
        return {
            "generation": "",
            "generation_error": {
                "error_type": type(error).__name__,
                "error_message": str(error),
            },
        }

    def _format_mini_swe_agent_output(self, trajectory_dict, data_point):
        trajectory_info = trajectory_dict["info"].copy()
        trajectory_info["model_name_or_path"] = self.cfg.server.model
        trajectory_info["instance_id"] = data_point["instance_id"]
        trajectory_info["generation"] = extract_final_answer(trajectory_info.pop("submission", None))
        return trajectory_info

    def _format_swe_agent_output(self, prediction_dict, data_point):
        trajectory_info = prediction_dict.copy()
        trajectory_info["model_name_or_path"] = self.cfg.server.model
        trajectory_info["instance_id"] = data_point["instance_id"]
        trajectory_info["generation"] = extract_final_answer(trajectory_info.pop("model_patch", None))
        return trajectory_info

    def _format_opencode_output(self, prediction_dict, data_point):
        trajectory_info = prediction_dict.copy()
        trajectory_info["model_name_or_path"] = self.cfg.server.model
        trajectory_info["instance_id"] = data_point["instance_id"]
        trajectory_info["generation"] = extract_final_answer(trajectory_info.pop("final_response", None))
        trajectory_info.pop("model_patch", None)
        return trajectory_info

    def _format_claude_code_output(self, prediction_dict, data_point):
        trajectory_info = prediction_dict.copy()
        trajectory_info["model_name_or_path"] = self.cfg.server.model
        trajectory_info["instance_id"] = data_point["instance_id"]
        trajectory_info["generation"] = extract_final_answer(trajectory_info.pop("final_response", None))
        trajectory_info.pop("model_patch", None)
        return trajectory_info

    def _openhands_requires_input_mount(self) -> bool:
        # The source Atlas row contains private grading data. OpenHands receives
        # a generated row containing only the fields required for inference.
        return False

    def _get_openhands_dataset_setup(self, data_point: dict, data_dir: str) -> str:
        safe_row = {
            "instance_id": data_point["instance_id"],
            "problem_statement": data_point["problem_statement"],
            "base_commit": data_point["base_commit"],
            "repo": data_point.get("repo", "swe-atlas-qna"),
            "version": data_point.get("version", data_point["instance_id"]),
            "container_repo_dir": data_point.get("container_repo_dir", "/testbed"),
            # The SWE-bench harness expects test lists even for Atlas Q&A tasks.
            "PASS_TO_PASS": [],
            "FAIL_TO_PASS": [],
        }
        serialized_row = json.dumps(safe_row, ensure_ascii=False)
        return (
            f"mkdir {shlex.quote(data_dir)} && "
            f"printf '%s\\n' {shlex.quote(serialized_row)} >{shlex.quote(data_dir + '/dataset.jsonl')}"
        )

    def _get_openhands_environment_setup(self) -> str:
        return (
            "/root/OpenHands/.venv/bin/python /nemo_run/code/nemo_skills/inference/eval/openhands_atlas.py "
            "--capture-environment /root/nemo-atlas-task-environment.json && "
        )

    def _get_openhands_prompt_setup(self, data_point: dict) -> str:
        extra_instructions = self._get_extra_instructions()
        template = """<uploaded_files>
{{ instance.container_repo_dir }}
</uploaded_files>

You are answering a software engineering question about the uploaded repository.
Inspect the repository with terminal commands and ground the answer in concrete code evidence.
Do not modify repository files. Do not implement a patch or create tests.

<question>
{{ instance.problem_statement }}
</question>

When you can answer the question confidently, stop exploring. Return the complete prose answer
inside two <<FINAL_ANSWER>> markers using the finish tool. Do not call tools after composing it."""
        if extra_instructions:
            template = f"{template}\n\n{extra_instructions}"
        template_name = "nemo_swe_atlas_qna.j2"
        template_path = f"evaluation/benchmarks/swe_bench/prompts/{template_name}"
        return (
            "python /nemo_run/code/nemo_skills/inference/eval/openhands_atlas.py "
            "--runner evaluation/benchmarks/swe_bench/run_infer.py && "
            f"printf '%s\\n' {shlex.quote(template)} >{shlex.quote(template_path)} && "
            f"export INSTRUCTION_TEMPLATE_NAME={shlex.quote(template_name)} && "
        )

    def _format_openhands_output(self, out_dict: dict, out_file: str, data_point: dict) -> str:
        final_response = extract_openhands_final_response(out_dict.get("history"))
        trajectory_info = out_dict.copy()
        trajectory_info["model_name_or_path"] = self.cfg.server.model
        trajectory_info["instance_id"] = data_point["instance_id"]
        trajectory_info["generation"] = extract_final_answer(final_response)
        trajectory_info["final_response"] = final_response
        # Atlas evaluates prose, never a patch. The complete raw OpenHands output
        # remains available in output.jsonl next to this normalized result.
        trajectory_info.pop("test_result", None)
        trajectory_info.pop("instance", None)

        prediction_file = str(Path(out_file).with_name("output_for_eval.jsonl"))
        with open(prediction_file, "w", encoding="utf-8") as fout:
            json.dump(trajectory_info, fout, ensure_ascii=False)
        return prediction_file

    def _get_claude_code_instruction(self, data_point):
        return (
            "You are a helpful assistant that can inspect a software repository to answer software engineering "
            "questions. Use Bash, Read, Glob, and Grep to gather concrete evidence, but do not modify repository "
            "files. Explain the relevant behavior with code references and observed evidence. When you are confident, "
            "return your complete prose answer wrapped in <<FINAL_ANSWER>> tags. Do not call tools after writing the "
            "final answer.\n\n"
            f"Question:\n{data_point['problem_statement']}"
        )

    def _get_extra_instructions_config_dir(self) -> str:
        return "eval/swe-atlas-qna/common"

    async def process_single_datapoint(self, data_point, data, prompt_format=None):
        api_base = self.get_api_base()

        async with self.semaphore:
            if self.cfg.agent_framework == SupportedAgentFrameworks.mini_swe_agent:
                output_file = await self._run_mini_swe_agent(data_point, api_base)
            elif self.cfg.agent_framework == SupportedAgentFrameworks.swe_agent:
                output_file = await self._run_swe_agent(data_point, api_base)
            elif self.cfg.agent_framework == SupportedAgentFrameworks.openhands:
                output_file = await self._run_openhands(data_point, api_base)
            elif self.cfg.agent_framework == SupportedAgentFrameworks.opencode:
                output_file = await self._run_opencode(data_point, api_base)
            elif self.cfg.agent_framework == SupportedAgentFrameworks.claude_code:
                output_file = await self._run_claude_code(data_point, api_base)
            else:
                raise ValueError(f"Unsupported agent framework: {self.cfg.agent_framework}")

        with open(output_file, "rt", encoding="utf-8") as fin:
            trajectory_info = json.load(fin)
        if self.cfg.agent_framework == SupportedAgentFrameworks.swe_agent:
            trajectory_info = self._format_swe_agent_output(trajectory_info, data_point)
        elif self.cfg.agent_framework == SupportedAgentFrameworks.opencode:
            trajectory_info = self._format_opencode_output(trajectory_info, data_point)
        elif self.cfg.agent_framework == SupportedAgentFrameworks.claude_code:
            trajectory_info = self._format_claude_code_output(trajectory_info, data_point)

        return {
            "generation": trajectory_info["generation"],
            "swe-atlas-qna-outputs": trajectory_info,
        }


GENERATION_TASK_CLASS = SweAtlasQnAGenerationTask


@hydra.main(version_base=None, config_name="base_swe_atlas_qna_generation_config")
def swe_atlas_qna_generation(cfg: SweAtlasQnAGenerationConfig):
    cfg = SweAtlasQnAGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    SweAtlasQnAGenerationTask(cfg).generate()


HELP_MESSAGE = get_help_message(
    SweAtlasQnAGenerationConfig,
    server_params=server_params(),
)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        swe_atlas_qna_generation()
