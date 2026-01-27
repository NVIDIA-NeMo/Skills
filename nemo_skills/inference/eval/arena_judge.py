# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import logging
import sys
from copy import deepcopy
from dataclasses import field

import hydra

from nemo_skills.inference.generate import (
    GenerationTask,
    GenerationTaskConfig,
    InferenceConfig,
)
from nemo_skills.inference.model import server_params
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    nested_dataclass,
    setup_logging,
)

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class ArenaJudgeConfig(GenerationTaskConfig):
    """Arena judge parameters.
    For the full list of supported parameters, use 'python -m nemo_skills.inference.generate --help'
    """

    # Inheritance was converting these dataclasses to dicts, so to be on the safe side we override them
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)

    # Override the default Generation config here
    prompt_config: str = "judge/arena"
    generation_key: str = "judgement"


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_arena_judge_config", node=ArenaJudgeConfig)


class ArenaJudgeTask(GenerationTask):
    def __init__(self, cfg: ArenaJudgeConfig):
        super().__init__(cfg)

    def log_example_prompt(self, all_data):
        data_point = deepcopy(all_data[0])

        if self.cfg.prompt_format == "openai":
            # print the prompt in openai format
            LOG.info("Example prompt in OpenAI format: \nData dictionary: %s", data_point)
            return

        data_point["answer_1"] = data_point["generation"]
        data_point["answer_2"] = data_point["baseline_answer"]
        LOG.info(
            "Example prompt:\nData dictionary: %s\nPrompt: %s", data_point, self.fill_prompt(data_point, all_data)
        )

    async def process_single_datapoint(self, data_point, all_data):
        gen_base_data = data_point.copy()
        gen_base_data["answer_1"] = data_point["generation"]
        gen_base_data["answer_2"] = data_point["baseline_answer"]
        # reversing the answers
        base_gen_data = data_point.copy()
        base_gen_data["answer_2"] = data_point["generation"]
        base_gen_data["answer_1"] = data_point["baseline_answer"]

        # Make two async calls instead of one batch call
        llm_output_1, llm_output_2 = await asyncio.gather(
            super().process_single_datapoint(gen_base_data, all_data),
            super().process_single_datapoint(base_gen_data, all_data),
        )

        return {
            f"{self.cfg.generation_key}-gen-base": llm_output_1["generation"],
            f"{self.cfg.generation_key}-base-gen": llm_output_2["generation"],
            "generation": "",  # dummy key since the downstream code expects it # TODO: fix this
        }


GENERATION_TASK_CLASS = ArenaJudgeTask


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name="base_arena_judge_config")
def generate(cfg: ArenaJudgeConfig):
    cfg = ArenaJudgeConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = ArenaJudgeTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    ArenaJudgeConfig,
    server_params=server_params(),
)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate()
