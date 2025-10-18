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
import json
import os
from abc import ABC
from typing import Any, Dict

import tqdm

from nemo_skills.utils import nested_dataclass


@nested_dataclass(kw_only=True)
class BaseEvaluatorConfig:
    # TODO: should we pass input_file separately everywhere?
    input_file: str | None = None  # could be None for interleaved evals
    data_dir: str | None = None
    split: str = "test"


class BaseEvaluator(ABC):
    """Base class for all evaluators."""

    def __init__(self, config: Dict[str, Any], num_parallel_requests=10):
        """Initialize evaluator with configuration."""
        self.config = config
        self.num_parallel_requests = num_parallel_requests

    async def eval_full(self) -> None:
        """Evaluate full dataset in batch mode."""
        semaphore = asyncio.Semaphore(self.num_parallel_requests)

        # assume that input_file is small enough to entirely fit in the memory
        async def process_line(line_data):
            # Concurrency control and merge updates into original record
            async with semaphore:
                updates = await self.eval_single(line_data)
                merged = dict(line_data)
                merged.update(updates)
                return merged

        with open(self.config.input_file, "rt", encoding="utf-8") as fin:
            tasks = []
            for file_line in fin:
                line_dict = json.loads(file_line)
                task = asyncio.create_task(process_line(line_dict))
                tasks.append(task)

        # Await tasks and write to temp file then replace original
        temp_file = self.config.input_file + "-tmp"
        with open(temp_file, "wt", encoding="utf-8") as f:
            for task in tqdm.tqdm(
                tasks, total=len(tasks), desc=f"Completed Evaluation for {os.path.basename(self.config.input_file)}"
            ):
                line = await task
                f.write(json.dumps(line) + "\n")

        # Replace original with temp file
        os.replace(temp_file, self.config.input_file)

    async def eval_single(self, data_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single data point during generation (optional).

        Args:
            data_point: Single data point with generation results

        Returns:
            Dict with evaluation results to merge into data_point

        Raises:
            NotImplementedError: If single evaluation is not supported
        """
        raise NotImplementedError(f"{self.__class__.__name__} does not support single evaluation during generation")

    def supports_single_eval(self) -> bool:
        """Check if this evaluator supports single evaluation during generation."""
        return self.__class__.eval_single is not BaseEvaluator.eval_single
