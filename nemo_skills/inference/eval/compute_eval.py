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
import logging
import sys

import hydra
from compute_eval.data.data_model import FileSolution

# noinspection PyProtectedMember
from compute_eval.generate_completions import _parse_solution

from nemo_skills.inference.generate import GenerateSolutionsConfig, GenerationTask
from nemo_skills.inference.model import server_params
from nemo_skills.utils import (
    get_help_message,
    get_logger_name,
    setup_logging,
)

_LOG = logging.getLogger(get_logger_name(__file__))


class ComputeEvalGenerationTask(GenerationTask):
    def __init__(self, cfg: GenerateSolutionsConfig):
        super().__init__(cfg)

    async def process_single_datapoint(self, data_point, data):
        res = await super().process_single_datapoint(data_point, data)
        try:
            solution = FileSolution(
                task_id=data_point["task_id"],
                files=_parse_solution(res["generation"]),
            )
            return {
                "solution": solution.model_dump(),
                "generation": res["generation"],
            }
        except KeyError as e:
            _LOG.error(f"Missing required field: {e}")
            return {
                "solution": None,
                "generation": res.get("generation", ""),
                "error": f"Missing required field: {e}",
            }
        except Exception as e:
            _LOG.error(f"Failed to parse solution: {e}")
            return {
                "solution": None,
                "generation": res.get("generation", ""),
                "error": f"Failed to parse solution: {e}",
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
