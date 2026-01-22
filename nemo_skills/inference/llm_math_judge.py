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

import logging
import sys
from dataclasses import field

import hydra

from nemo_skills.evaluation.math_grader import extract_answer
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
    prefill_judgement,
    setup_logging,
)

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class LlmMathJudgeConfig(GenerationTaskConfig):
    """LLM math judge parameters.
    For the full list of supported parameters, use 'python -m nemo_skills.inference.generate --help'
    """

    # Inheritance was converting these dataclasses to dicts, so to be on the safe side we override them
    inference: InferenceConfig = field(default_factory=InferenceConfig)  # LLM call parameters
    # Inference server configuration {server_params}
    server: dict = field(default_factory=dict)

    # Override the default Generation config here
    prompt_config: str = "judge/math"
    generation_key: str = "judgement"

    add_generation_stats: bool = False


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="base_llm_math_judge_config", node=LlmMathJudgeConfig)


class LLMMathJudgeTask(GenerationTask):
    def __init__(self, cfg: LlmMathJudgeConfig):
        super().__init__(cfg)

    def preprocess_data(self, data):
        """Extract the predicted answer from the generation."""
        for data_point in data:
            if "predicted_answer" not in data_point:
                data_point["predicted_answer"] = extract_answer(data_point["generation"])

        return data

    def prefill_generation(self, data_point):
        """Prefill judgement"""
        judgement = prefill_judgement(data_point)
        if judgement is None:
            return None
        else:
            return {"generation": judgement}


GENERATION_TASK_CLASS = LLMMathJudgeTask


# Update the hydra main to use the class method
@hydra.main(version_base=None, config_name="base_llm_math_judge_config")
def generate(cfg: LlmMathJudgeConfig):
    cfg = LlmMathJudgeConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)

    task = LLMMathJudgeTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    LlmMathJudgeConfig,
    server_params=server_params(),
)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate()
