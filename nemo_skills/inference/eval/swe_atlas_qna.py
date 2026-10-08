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

import json
import logging
import shlex
import sys
from pathlib import Path

import hydra

from nemo_skills.inference.eval.swebench import (
    SupportedAgentFrameworks,
    SweBenchGenerationConfig,
    SweBenchGenerationTask,
)
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))

FINAL_ANSWER_TAG = "<<FINAL_ANSWER>>"
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


class SweAtlasQnAGenerationTask(SweBenchGenerationTask):
    """Run a repository agent and return its submitted prose answer."""

    def __init__(self, cfg: SweBenchGenerationConfig):
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
        }
        serialized_row = json.dumps(safe_row, ensure_ascii=False)
        return (
            f"mkdir {shlex.quote(data_dir)} && "
            f"printf '%s\\n' {shlex.quote(serialized_row)} >{shlex.quote(data_dir + '/dataset.jsonl')}"
        )

    def _get_openhands_prompt_setup(self, data_point: dict) -> str:
        extra_instructions = self._get_extra_instructions()
        template = """<uploaded_files>
/workspace/{{ workspace_dir_name }}
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


@hydra.main(version_base=None, config_name="base_swebench_generation_config")
def swe_atlas_qna_generation(cfg: SweBenchGenerationConfig):
    cfg = SweBenchGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    SweAtlasQnAGenerationTask(cfg).generate()


HELP_MESSAGE = get_help_message(
    SweBenchGenerationConfig,
    server_params=server_params(),
)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        swe_atlas_qna_generation()
