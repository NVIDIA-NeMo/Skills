# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Annotated

import hydra
from compute_eval.data.data_model import CudaCppProblem, CudaPythonProblem
from compute_eval.generate_completions import generate_model_completions
from compute_eval.prompts import SYSTEM_PROMPT
from pydantic import Field, TypeAdapter

from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask
from nemo_skills.inference.model import server_params
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    setup_logging,
)

_LOG = logging.getLogger(get_logger_name(__file__))
_TYPE_ADAPTER = TypeAdapter(Annotated[CudaCppProblem | CudaPythonProblem, Field(discriminator="type")])


class ComputeEvalGenerationTask(GenerationTask):
    _system_prompt: str
    _model: str

    def __init__(self, cfg: GenerateSolutionsConfig):
        super().__init__(cfg)
        self._system_prompt = cfg.system_message or SYSTEM_PROMPT
        self._model = cfg.server.get("model", "")
        if self._model == "":
            _LOG.error("Model must be specified in server configuration.")
            raise RuntimeError

        self._base_url = cfg.server.get("base_url", "")
        if self._base_url == "":
            _LOG.error("Base URL must be specified in server configuration.")
            raise RuntimeError

    def log_example_prompt(self, data):
        return

    def setup_prompt(self):
        return

    def setup_llm(self):
        return

    def setup_litellm_cache(self):
        return

    def cleanup_litellm_cache(self):
        return

    async def process_single_datapoint(self, data_point, data):
        problem = _TYPE_ADAPTER.validate_python(data_point["problem"])
        solution = await asyncio.to_thread(
            generate_model_completions,
            system_prompt=self._system_prompt,
            problem=problem,
            model=self._model,
            base_url=self._base_url,
            params=None,
        )

        return {
            "solution": solution.model_dump(),
            "generation": "",
        }


GENERATION_TASK_CLASS = ComputeEvalGenerationTask


@hydra.main(version_base=None, config_name="base_generation_config")
def run_compute_eval(cfg: GenerateSolutionsConfig):
    _LOG.info("Config used: %s", cfg)

    task = ComputeEvalGenerationTask(cfg)
    task.generate()


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(get_help_message(GenerateSolutionsConfig, server_params=server_params()))
    else:
        setup_logging()
        run_compute_eval()
