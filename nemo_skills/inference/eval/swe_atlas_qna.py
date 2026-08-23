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
import sys

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

    async def process_single_datapoint(self, data_point, data, prompt_format=None):
        api_base = self.get_api_base()

        async with self.semaphore:
            if self.cfg.agent_framework == SupportedAgentFrameworks.mini_swe_agent:
                output_file = await self._run_mini_swe_agent(data_point, api_base)
            elif self.cfg.agent_framework == SupportedAgentFrameworks.swe_agent:
                output_file = await self._run_swe_agent(data_point, api_base)
            else:
                raise ValueError(f"Unsupported agent framework: {self.cfg.agent_framework}")

        with open(output_file, "rt", encoding="utf-8") as fin:
            trajectory_info = json.load(fin)
        if self.cfg.agent_framework == SupportedAgentFrameworks.swe_agent:
            trajectory_info = self._format_swe_agent_output(trajectory_info, data_point)

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
