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
from typing import Annotated, Any

from compute_eval.data.data_model import CudaCppProblem, CudaPythonProblem, FileSolution, PatchSolution
from compute_eval.execution import evaluate_solution
from compute_eval.utils.eval_utils import get_nvcc_version, parse_semver
from pydantic import Field, TypeAdapter

from nemo_skills.evaluation.evaluator import BaseEvaluator
from nemo_skills.utils import get_logger_name

_LOG = logging.getLogger(get_logger_name(__file__))
_PROBLEM_ADAPTER = TypeAdapter(Annotated[CudaCppProblem | CudaPythonProblem, Field(discriminator="type")])
_SOLUTION_ADAPTER = TypeAdapter(Annotated[FileSolution | PatchSolution, Field(discriminator="type")])


class ComputeEvalEvaluator(BaseEvaluator):
    _installed_ctk_major: int
    _installed_ctk_minor: int

    def __init__(self, config: dict, num_parallel_requests=10):
        super().__init__(config, num_parallel_requests)
        nvcc_version = get_nvcc_version()
        if not nvcc_version:
            raise RuntimeError(
                "NVCC not found. Please ensure that the CUDA Toolkit is installed and nvcc is in your PATH."
            )

        self._installed_ctk_major, self._installed_ctk_minor, _ = parse_semver(nvcc_version)

    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        # noinspection PyBroadException
        try:
            problem = _PROBLEM_ADAPTER.validate_python(data_point["problem"])
            solution = _SOLUTION_ADAPTER.validate_python(data_point["solution"])

            graded = await asyncio.to_thread(
                evaluate_solution,
                installed_ctk_major=self._installed_ctk_major,
                installed_ctk_minor=self._installed_ctk_minor,
                problem=problem,
                solution=solution,
            )

            return {
                "passed": graded.passed,
                "skipped": graded.skipped,
                "elapsed_time": graded.elapsed_time,
                "build_output": graded.build_output,
                "test_output": graded.test_output,
            }
        except KeyError as e:
            _LOG.error(f"Missing required field in data_point: {e}")
            return {
                "passed": False,
                "skipped": False,
                "elapsed_time": 0.0,
                "build_output": "",
                "test_output": "",
                "error": f"Missing required field: {e}",
            }
        except Exception as e:
            _LOG.error(f"Error during evaluation: {e}")
            return {
                "passed": False,
                "skipped": False,
                "elapsed_time": 0.0,
                "build_output": "",
                "test_output": "",
                "error": str(e),
            }
