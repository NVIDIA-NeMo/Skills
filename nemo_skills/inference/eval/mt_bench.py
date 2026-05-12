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

import hydra

from nemo_skills.code_execution.sandbox import sandbox_params
from nemo_skills.inference.generate import GenerationTask, GenerationTaskConfig
from nemo_skills.inference.model import server_params
from nemo_skills.inference.model.base import EndpointType
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, parse_reasoning, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class MTBenchInferenceConfig:
    endpoint_type: EndpointType = EndpointType.chat
    temperature: float = 0.0
    top_k: int | None = -1
    top_p: float | None = None
    min_p: float | None = 0.0
    random_seed: int | None = None
    tokens_to_generate: int | None = None
    repetition_penalty: float | None = None
    top_logprobs: int | None = None
    timeout: int | None = 14400
    reasoning_effort: str | None = None
    extra_body: dict = field(default_factory=dict)


@nested_dataclass(kw_only=True)
class MTBenchGenerationConfig(GenerationTaskConfig):
    """MT-Bench two-turn generation config."""

    inference: MTBenchInferenceConfig = field(default_factory=MTBenchInferenceConfig)
    server: dict = field(default_factory=dict)
    prompt_format: str = "openai"


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_mt_bench_generation_config", node=MTBenchGenerationConfig)


class MTBenchGenerationTask(GenerationTask):
    """Two-turn generation for MT-Bench: generate answer_1, then answer_2."""

    def fill_prompt(self, data_point, data, prompt_format=None):
        if "messages" in data_point:
            return data_point["messages"]
        return [{"role": "user", "content": data_point["question_1"]}]

    async def process_single_datapoint(self, data_point, all_data):
        """Process a single datapoint with two-turn generation.

        The flow is:
        1. Generate answer for question 1
        2. Generate answer for question 2 conditioned on answer 1
        """
        # ===== Turn 1: Generate answer for question 1 =====
        turn1_result = await super().process_single_datapoint(data_point, all_data)
        LOG.debug("Turn 1 result: %s", turn1_result)

        if self.cfg.parse_reasoning:
            parse_reasoning(turn1_result, self.cfg.generation_key, self.cfg.end_reasoning_string)

        answer_1 = turn1_result[self.cfg.generation_key]
        LOG.debug("Answer 1: %s", answer_1)

        # ===== Turn 2: Generate answer for question 2 =====
        turn2_messages = [
            {"role": "user", "content": data_point["question_1"]},
            {"role": "assistant", "content": answer_1},
            {"role": "user", "content": data_point["question_2"]},
        ]
        LOG.debug("Turn 2 messages: %s", turn2_messages)

        turn2_data_point = {"messages": turn2_messages}
        turn2_result = await super().process_single_datapoint(turn2_data_point, all_data)
        LOG.debug("Turn 2 result: %s", turn2_result)

        raw_turn2_generation = turn2_result[self.cfg.generation_key]
        if self.cfg.parse_reasoning:
            parse_reasoning(turn2_result, self.cfg.generation_key, self.cfg.end_reasoning_string)

        answer_2 = turn2_result[self.cfg.generation_key]

        return {
            "answer_1": answer_1,
            "answer_2": answer_2,
            "generation": raw_turn2_generation,
            "num_generated_tokens": (
                turn1_result.get("num_generated_tokens", 0) + turn2_result.get("num_generated_tokens", 0)
            ),
        }


GENERATION_TASK_CLASS = MTBenchGenerationTask


@hydra.main(version_base=None, config_name="base_mt_bench_generation_config")
def generate(cfg: MTBenchGenerationConfig):
    cfg = MTBenchGenerationConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    task = MTBenchGenerationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    MTBenchGenerationConfig,
    server_params=server_params(),
    sandbox_params=sandbox_params(),
)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate()
