# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import re
import sys
from dataclasses import asdict, field, is_dataclass
from typing import List

import hydra

from nemo_skills.code_execution.sandbox import sandbox_params
from nemo_skills.inference.generate import GenerationTask, GenerationTaskConfig
from nemo_skills.inference.model import server_params
from nemo_skills.inference.model.base import EndpointType
from nemo_skills.prompt.utils import get_token_count, load_config
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    nested_dataclass,
    parse_reasoning,
    setup_logging,
)

LOG = logging.getLogger(get_logger_name(__file__))


def parse_chatml(prompt: str) -> List[dict]:
    pattern = re.compile(r"<\|start\|>(.*?)<\|message\|>(.*?)<\|end\|>", re.DOTALL)

    messages = []
    for role, content in pattern.findall(prompt):
        messages.append({"role": role.strip(), "content": content.strip()})
    return messages


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

    prompt_config: str = "eval/critpt/solve_problem"
    prompt_config_turn2: str = "eval/critpt/code_output"


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_critpt_generation_config", node=CritPtGenerationConfig)


class CritPtGenerationTask(GenerationTask):
    """Custom generation task for CritPt benchmark with two-turn conversation."""

    def __init__(self, cfg: GenerationTaskConfig):
        super().__init__(cfg)

    async def _process_single_completion(self, data_point, filled_prompt):
        # Handle inference config - check if it's a dataclass or already a dict
        if is_dataclass(self.cfg.inference):
            inference_params = asdict(self.cfg.inference)
        else:
            # Already a dict from Hydra
            inference_params = dict(self.cfg.inference)

        generation_params = {
            **inference_params,
            **self.extra_generate_params,
            "prompt": filled_prompt,
            "stop_phrases": [self.cfg.stop_phrase] if self.cfg.stop_phrase else None,
        }

        if self.cfg.code_execution:
            if self.cfg.override_max_code_executions and self.cfg.total_code_executions_in_prompt is not None:
                generation_params["max_code_executions"] = data_point["total_code_executions"]

        result = await self.generate_with_semaphore(**generation_params)

        if self.cfg.count_prompt_tokens:
            num_input_tokens = get_token_count(self.hf_tokenizer, generation_params["prompt"])
            result["num_input_tokens"] = num_input_tokens

        return result

    async def process_single_datapoint(self, data_point, all_data):
        """Process a single datapoint with two-turn generation.

        The flow is:
        1. Generate solution for the problem
        2. Generate code implementation using the solution + code template
        """
        # ===== Turn 1: Generate solution for the problem =====
        # In this round, we still use the NS prompt filling pipeline.

        # Since process_single_datapoint did not return the prompt, we fill it here manually.
        # turn1_prompt contains the system message, and the user message.
        turn1_prompt = self.fill_prompt(data_point, all_data)
        LOG.debug(f"Turn 1 prompt: {turn1_prompt}")

        if isinstance(turn1_prompt, str):
            turn1_prompt = parse_chatml(turn1_prompt)
            LOG.debug(f"Turn 1 prompt (parsed): {turn1_prompt}")

        turn1_result = await super().process_single_datapoint(data_point, all_data)
        LOG.debug(f"Turn 1 result: {turn1_result}")
        #
        if self.cfg.end_reasoning_string in turn1_result[self.cfg.generation_key]:
            parse_reasoning(turn1_result, self.cfg.generation_key, self.cfg.end_reasoning_string)
        solution_turn1 = turn1_result[self.cfg.generation_key]
        LOG.debug(f"Solution: {solution_turn1}")

        assistant_msg: dict = {"role": "assistant", "content": solution_turn1}

        LOG.debug(f"Assistant: {assistant_msg}")

        # ===== Turn 2: Generate code using template =====
        # Build prompt that includes code template
        turn2_prompt_template = load_config(self.cfg.prompt_config_turn2)["user"]
        turn2_prompt: dict = {"role": "user", "content": turn2_prompt_template.format(**data_point)}
        LOG.debug(f"Turn 2 prompt: {turn2_prompt}")

        # Final prompt is the turn1_prompt + assistant message + turn2_prompt
        final_prompt = turn1_prompt + [assistant_msg, turn2_prompt]

        turn2_result = await self._process_single_completion(data_point, final_prompt)

        if self.cfg.end_reasoning_string in turn2_result[self.cfg.generation_key]:
            parse_reasoning(turn2_result, self.cfg.generation_key, self.cfg.end_reasoning_string)

        turn2_result["intermediate"] = solution_turn1
        turn2_result["num_generated_tokens_turn1"] = turn1_result.get("num_generated_tokens", 0)
        turn2_result["num_generated_tokens_turn2"] = turn2_result.get("num_generated_tokens", 0)
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
