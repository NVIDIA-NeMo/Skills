# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import logging
import sys
from dataclasses import field
from typing import List

import hydra

from nemo_skills.code_execution.sandbox import sandbox_params
from nemo_skills.inference.generate import GenerationTask, GenerationTaskConfig
from nemo_skills.inference.model import server_params
from nemo_skills.inference.model.base import EndpointType
from nemo_skills.prompt.utils import get_prompt, load_config
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    nested_dataclass,
    parse_reasoning,
    setup_logging,
)

LOG = logging.getLogger(get_logger_name(__file__))


# Like nemo_skills.inference.generate.InferenceConfig, except most parameters are not passed by default
# because they may not be supported by all LLM servers.
@nested_dataclass(kw_only=True)
class CritPtInferenceConfig:
    endpoint_type: EndpointType = EndpointType.chat
    temperature: float = 0.0  # Temperature of 0 means greedy decoding
    top_k: int | None = -1
    top_p: float | None = None
    min_p: float | None = 0.0
    random_seed: int | None = None
    tokens_to_generate: int | None = None
    repetition_penalty: float | None = None
    top_logprobs: int | None = None
    timeout: int | None = 14400  # Timeout for each individual LLM call in seconds
    reasoning_effort: str | None = None
    extra_body: dict = field(default_factory=dict)  # Any other extra params passed with extra_body argument


@nested_dataclass(kw_only=True)
class CritPtGenerationConfig(GenerationTaskConfig):
    """CritPt benchmark generation with two-turn conversation.
    For the full list of supported parameters, use 'python -m nemo_skills.inference.generate --help'
    """

    # Inheritance was converting these dataclasses to dicts, so to be on the safe side we override them
    inference: CritPtInferenceConfig = field(default_factory=CritPtInferenceConfig)  # LLM call parameters
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)

    # Use openai format to work directly with messages list
    prompt_format: str = "openai"
    prompt_config_turn1: str = "eval/critpt/solve_problem"
    prompt_config_turn2: str = "eval/critpt/code_output"


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_critpt_generation_config", node=CritPtGenerationConfig)


class CritPtGenerationTask(GenerationTask):
    """Custom generation task for CritPt benchmark with two-turn conversation."""

    def __init__(self, cfg: GenerationTaskConfig):
        super().__init__(cfg)
        # Load prompt templates for both turns
        self.prompt_config_turn1 = load_config(self.cfg.prompt_config_turn1)
        self.prompt_config_turn2 = load_config(self.cfg.prompt_config_turn2)
        # Pre-load prompt instance to avoid repeated calls
        self.turn1_prompt_instance = get_prompt(prompt_config=self.prompt_config_turn1)
        self.turn2_prompt_instance = get_prompt(prompt_config=self.prompt_config_turn2)

    def fill_prompt(self, data_point, data):
        """Build messages list for turn 1, or return pre-built messages for turn 2."""
        # If messages are already provided (turn 2), just return them
        if "messages" in data_point:
            return data_point["messages"]

        # Turn 1: Build messages from prompt config
        turn1_messages = self.turn1_prompt_instance.fill(input_dict=data_point)

        return turn1_messages

    async def process_single_datapoint(self, data_point, all_data):
        """Process a single datapoint with two-turn generation.

        The flow is:
        1. Generate solution for the problem (turn 1)
        2. Generate code implementation using the solution + code template (turn 2)
        """
        # ===== Turn 1: Generate solution for the problem =====
        # fill_prompt will be called by super() and will build messages for turn 1
        turn1_result = await super().process_single_datapoint(data_point, all_data)
        LOG.debug(f"Turn 1 result: {turn1_result}")

        if self.cfg.parse_reasoning:
            parse_reasoning(turn1_result, self.cfg.generation_key, self.cfg.end_reasoning_string)

        solution_turn1 = turn1_result[self.cfg.generation_key]
        LOG.debug(f"Solution: {solution_turn1}")

        # Get the turn1 messages by calling fill_prompt (it will rebuild them)
        turn1_messages = self.fill_prompt(data_point, all_data)

        # ===== Turn 2: Generate code using template =====
        # Build messages for turn 2: turn1_messages + assistant response + turn2_user_message
        turn2_user_messages: List[dict] = self.turn2_prompt_instance.fill(input_dict=data_point)

        turn2_messages = turn1_messages + [{"role": "assistant", "content": solution_turn1}] + turn2_user_messages
        LOG.debug(f"Turn 2 messages: {turn2_messages}")

        # Use a data point with turn 2 messages
        # fill_prompt will detect the "messages" key and return them directly
        turn2_data_point = {"messages": turn2_messages}
        turn2_result = await super().process_single_datapoint(turn2_data_point, all_data)
        LOG.debug(f"Turn 2 result: {turn2_result[self.cfg.generation_key]}")

        # Add turn-specific metadata
        turn2_result["intermediate"] = solution_turn1
        turn2_result["num_generated_tokens_turn1"] = turn1_result["num_generated_tokens"]
        turn2_result["num_generated_tokens_turn2"] = turn2_result["num_generated_tokens"]

        return turn2_result


GENERATION_TASK_CLASS = CritPtGenerationTask


@hydra.main(version_base=None, config_name="base_critpt_generation_config")
def generate(cfg: CritPtGenerationConfig):
    cfg = CritPtGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = CritPtGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    CritPtGenerationConfig,
    server_params=server_params(),
    sandbox_params=sandbox_params(),
)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate()
