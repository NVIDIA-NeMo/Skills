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

import asyncio
import logging
import sys
from dataclasses import field

import hydra

from nemo_skills.inference.generate import GenerationTask, GenerationTaskConfig, InferenceConfig
from nemo_skills.inference.model import server_params
from nemo_skills.prompt.utils import get_prompt, load_config
from nemo_skills.utils import get_help_message, get_logger_name, nested_dataclass, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))

NEED_REF_CATS = {"math", "reasoning", "coding"}
JUDGE_TIMEOUT = 180  # 3 minutes per individual judge API call


@nested_dataclass(kw_only=True)
class MTBenchJudgeConfig(GenerationTaskConfig):
    """MT-Bench judge config with per-turn prompt templates."""

    inference: InferenceConfig = field(default_factory=InferenceConfig)
    server: dict = field(default_factory=dict)

    prompt_format: str = "openai"
    generation_key: str = "judgement"

    prompt_config_turn1: str = "judge/mt-bench/turn1"
    prompt_config_turn1_with_ref: str = "judge/mt-bench/turn1_with_ref"
    prompt_config_turn2: str = "judge/mt-bench/turn2"
    prompt_config_turn2_with_ref: str = "judge/mt-bench/turn2_with_ref"

    num_judges: int = 1  # Number of judge calls per turn (averaged to reduce variance)


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_mt_bench_judge_config", node=MTBenchJudgeConfig)


class MTBenchJudgeTask(GenerationTask):
    """Judge task for MT-Bench: score each turn on 1-10 scale."""

    def __init__(self, cfg: MTBenchJudgeConfig):
        super().__init__(cfg)
        self.prompt_turn1 = get_prompt(prompt_config=load_config(self.cfg.prompt_config_turn1))
        self.prompt_turn1_with_ref = get_prompt(prompt_config=load_config(self.cfg.prompt_config_turn1_with_ref))
        self.prompt_turn2 = get_prompt(prompt_config=load_config(self.cfg.prompt_config_turn2))
        self.prompt_turn2_with_ref = get_prompt(prompt_config=load_config(self.cfg.prompt_config_turn2_with_ref))

    def fill_prompt(self, data_point, data, prompt_format=None):
        if "messages" in data_point:
            return data_point["messages"]
        # Build judge prompt for preview (dry-run); answers may not exist yet.
        category = data_point["category"]
        prompt = self._select_prompt(category, turn=1)
        judge_input = {
            "question_1": data_point["question_1"],
            "answer_1": data_point.get("answer_1", "(not yet generated)"),
        }
        if category in NEED_REF_CATS:
            judge_input["ref_answer_1"] = data_point["ref_answer_1"]
        return prompt.fill(input_dict=judge_input)

    def log_example_prompt(self, data):
        if data:
            LOG.info("Example judge prompt:\n%s", self.fill_prompt(data[0], data))

    def _select_prompt(self, category, turn):
        has_ref = category in NEED_REF_CATS
        if turn == 1:
            return self.prompt_turn1_with_ref if has_ref else self.prompt_turn1
        else:
            return self.prompt_turn2_with_ref if has_ref else self.prompt_turn2

    async def _judge_turn(self, data_point, all_data, turn):
        category = data_point["category"]
        prompt = self._select_prompt(category, turn)

        if turn == 1:
            judge_input = {
                "question_1": data_point["question_1"],
                "answer_1": data_point["answer_1"],
            }
            if category in NEED_REF_CATS:
                judge_input["ref_answer_1"] = data_point["ref_answer_1"]
        else:
            judge_input = {
                "question_1": data_point["question_1"],
                "answer_1": data_point["answer_1"],
                "question_2": data_point["question_2"],
                "answer_2": data_point["answer_2"],
            }
            if category in NEED_REF_CATS:
                judge_input["ref_answer_1"] = data_point["ref_answer_1"]
                judge_input["ref_answer_2"] = data_point["ref_answer_2"]

        messages = prompt.fill(input_dict=judge_input)
        judge_data_point = {"messages": messages}
        result = await super().process_single_datapoint(judge_data_point, all_data)
        return result["generation"]

    async def _safe_judge(self, data_point, all_data, turn):
        """Call _judge_turn with a timeout to prevent indefinite hangs."""
        try:
            return await asyncio.wait_for(
                self._judge_turn(data_point, all_data, turn),
                timeout=JUDGE_TIMEOUT,
            )
        except asyncio.TimeoutError:
            LOG.warning(
                "Judge call timed out (turn=%d, timeout=%ds, category=%s)",
                turn,
                JUDGE_TIMEOUT,
                data_point["category"],
            )
            return ""

    async def process_single_datapoint(self, data_point, all_data, prompt_format=None):
        # Skip judging if generation failed (answer fields missing)
        if "answer_1" not in data_point or "answer_2" not in data_point:
            LOG.warning("Skipping judge for data point missing answer_1/answer_2 (generation likely failed)")
            return {
                "judgement_turn1": "",
                "judgement_turn2": "",
                "generation": "",
                "error": "missing_answers",
            }

        num_judges = self.cfg.num_judges

        if num_judges <= 1:
            judgement_turn1, judgement_turn2 = await asyncio.gather(
                self._safe_judge(data_point, all_data, turn=1),
                self._safe_judge(data_point, all_data, turn=2),
            )
            return {
                "judgement_turn1": judgement_turn1,
                "judgement_turn2": judgement_turn2,
                "generation": "",
            }

        # Multiple judges: call N times in parallel and store all judgements
        tasks = []
        for _ in range(num_judges):
            tasks.append(self._safe_judge(data_point, all_data, turn=1))
            tasks.append(self._safe_judge(data_point, all_data, turn=2))
        results = await asyncio.gather(*tasks)

        # results = [turn1_j0, turn2_j0, turn1_j1, turn2_j1, ...]
        judgements_turn1 = [results[i] for i in range(0, len(results), 2)]
        judgements_turn2 = [results[i] for i in range(1, len(results), 2)]

        return {
            "judgement_turn1": judgements_turn1[0],  # primary (for backward compat)
            "judgement_turn2": judgements_turn2[0],
            "all_judgements_turn1": judgements_turn1,
            "all_judgements_turn2": judgements_turn2,
            "num_judges": num_judges,
            "generation": "",
        }


GENERATION_TASK_CLASS = MTBenchJudgeTask


@hydra.main(version_base=None, config_name="base_mt_bench_judge_config")
def generate(cfg: MTBenchJudgeConfig):
    cfg = MTBenchJudgeConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    task = MTBenchJudgeTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    MTBenchJudgeConfig,
    server_params=server_params(),
)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate()
