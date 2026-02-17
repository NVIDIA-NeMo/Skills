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
from nemo_skills.inference.model.base import EndpointType
from nemo_skills.prompt.utils import get_prompt
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
    # prompt_config is used as the default for any category not explicitly mapped below
    prompt_config: str = "judge/arena"
    generation_key: str = "judgement"

    # Category-specific prompt config overrides (arena-hard-v2 uses different prompts per category)
    # Set to None to use the default prompt_config for that category
    # creative_writing uses a prompt that doesn't ask the judge to generate its own answer first
    prompt_config_creative: str = "judge/arena_creative"


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_arena_judge_config", node=ArenaJudgeConfig)


class ArenaJudgeTask(GenerationTask):
    def __init__(self, cfg: ArenaJudgeConfig):
        super().__init__(cfg)

    def setup_prompt(self):
        if self.cfg.prompt_format == "openai":
            return None

        # Load the default prompt (used for most categories including hard_prompt, arena-hard-v0.1, etc.)
        default_prompt = get_prompt(
            prompt_config=self.cfg.prompt_config,
            tokenizer=self.tokenizer,
            code_tags=self.cfg.code_tags,
            examples_type=self.cfg.examples_type,
            system_message=self.cfg.system_message,
        )

        # Load category-specific prompt overrides
        self.category_prompts = {}
        if self.cfg.prompt_config_creative:
            self.category_prompts["creative_writing"] = get_prompt(
                prompt_config=self.cfg.prompt_config_creative,
                tokenizer=self.tokenizer,
                code_tags=self.cfg.code_tags,
                examples_type=self.cfg.examples_type,
                system_message=self.cfg.system_message,
            )
            LOG.info("Prompt used (creative_writing): %s", self.category_prompts["creative_writing"])
        # registering default prompt explicitly for hard_prompt
        self.category_prompts["hard_prompt"] = default_prompt

        LOG.info("Prompt used (default): %s", default_prompt)
        return default_prompt

    def fill_prompt(self, data_point, data, prompt_format=None):
        """Fill prompt with category-specific prompt config."""
        prompt_format = prompt_format or self.cfg.prompt_format
        if prompt_format == "openai":
            return super().fill_prompt(data_point=data_point, data=data, prompt_format=prompt_format)

        # Select the appropriate prompt based on category. If not defined, forcing fall-back to default prompt
        category = data_point.get("category")
        if not category:
            prompt = self.prompt
        else:
            # will fail if category not in category_prompts as this is unexpected
            prompt = self.category_prompts[category]

        data_point = deepcopy(data_point)
        filled_prompt = prompt.fill(
            data_point,
            start_assistant_response_key=self.cfg.start_assistant_response_key,
            chat_template_kwargs=self.cfg.chat_template_kwargs,
            format_as_string=(self.cfg.inference.endpoint_type == EndpointType.text),
        )
        if self.cfg.prompt_suffix:
            if isinstance(filled_prompt, list):
                filled_prompt[-1]["content"] += self.cfg.prompt_suffix
            else:
                filled_prompt += self.cfg.prompt_suffix
        return filled_prompt

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

    async def process_single_datapoint(self, data_point, all_data, prompt_format=None):
        gen_base_data = data_point.copy()
        gen_base_data["answer_1"] = data_point["generation"]
        gen_base_data["answer_2"] = data_point["baseline_answer"]
        # reversing the answers
        base_gen_data = data_point.copy()
        base_gen_data["answer_2"] = data_point["generation"]
        base_gen_data["answer_1"] = data_point["baseline_answer"]

        # Make two async calls instead of one batch call
        llm_output_1, llm_output_2 = await asyncio.gather(
            super().process_single_datapoint(gen_base_data, all_data, prompt_format),
            super().process_single_datapoint(base_gen_data, all_data, prompt_format),
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
