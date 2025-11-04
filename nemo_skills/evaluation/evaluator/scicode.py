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
import logging
import re
from dataclasses import field

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.evaluation.evaluator.base import BaseEvaluatorConfig
from nemo_skills.inference.eval.scicode_utils import eval_prefix
from nemo_skills.utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class ScicodeEvaluatorConfig(BaseEvaluatorConfig):
    sandbox: dict = field(default_factory=lambda: {"sandbox_type": "local"})
    timeout: float = 30.0
    num_parallel_requests: int = 20


async def _execute_single_test(args):
    """Helper function to execute a single test case."""
    eval_config, elem_idx, full_generation, json_content, subtask_step = args

    # step_id is always problem_id.subtask_step
    step_number = json_content["sub_steps"][int(subtask_step) - 1]["step_number"]
    test_lst = json_content["sub_steps"][int(subtask_step) - 1]["test_cases"]
    # Remove any external scicode import lines; we provide local helpers in eval_prefix
    sanitized_tests = []
    for tc in test_lst:
        lines = []
        for line in tc.split("\n"):
            if re.match(r"^\s*(from\s+scicode\.|import\s+scicode)\b", line):
                continue
            lines.append(line)
        sanitized_tests.append("\n".join(lines))

    code = full_generation + eval_prefix + f"targets = process_hdf5_to_tuple('{step_number}', {len(test_lst)})\n"
    for idx in range(len(sanitized_tests)):
        code += f"target = targets[{idx}]\n\n"
        for line in sanitized_tests[idx].split("\n"):
            code += line + "\n"

    sandbox = get_sandbox(**eval_config.sandbox)
    output_dict, _ = await sandbox.execute_code(code, timeout=eval_config.timeout, max_output_characters=100000)

    return elem_idx, output_dict


def test_code(eval_config, scicode_data):
    # adapted from https://github.com/scicode-bench/SciCode/blob/main/eval/scripts/test_generated_code.py
    json_idx = {}

    for prob_data in scicode_data:
        json_idx[prob_data["problem_id"]] = scicode_data.index(prob_data)

    # Prepare all tasks for parallel execution
    tasks = []
    for elem_idx, elem in enumerate(scicode_data):
        for step_id, full_generation in elem["generation"].items():
            problem_id, subtask_step = step_id.split(".")
            json_content = scicode_data[json_idx[problem_id]]
            tasks.append((eval_config, elem_idx, full_generation, json_content, subtask_step))

    # Initialize status_lists with correct structure
    status_lists = [[] for _ in range(len(scicode_data))]

    # Execute tasks in parallel using asyncio with a semaphore for num_parallel_requests
    async def run_tasks():
        semaphore = asyncio.Semaphore(eval_config.num_parallel_requests)

        async def execute_with_semaphore(task_args):
            async with semaphore:
                return await _execute_single_test(task_args)

        return await asyncio.gather(*[execute_with_semaphore(task) for task in tasks])

    results = asyncio.run(run_tasks())

    # Organize results back into the original structure
    for elem_idx, output_dict in results:
        status_lists[elem_idx].append(output_dict)

    return status_lists


def eval_scicode(cfg):
    eval_config = ScicodeEvaluatorConfig(**cfg)

    # Install required packages for scicode evaluation
    LOG.info("Installing required packages and data for scicode evaluation...")

    async def install_packages():
        sandbox = get_sandbox(**eval_config.sandbox)

        # Install scipy 1.10.1 - the last version that includes scipy.integrate.simps
        # which is used by some scicode tests. Newer versions renamed it to simpson.
        result, _ = await sandbox.execute_code(
            "pip install scipy==1.10.1 --force-reinstall -q", language="shell", timeout=120.0
        )
        if result["process_status"] != "completed":
            LOG.warning(f"Failed to install scipy 1.10.1: {result.get('stderr', 'Unknown error')}")
        else:
            LOG.info("Successfully installed scipy 1.10.1")

        # Upgrade matplotlib for tests that use mpl_toolkits.mplot3d
        result, _ = await sandbox.execute_code("pip install --upgrade matplotlib -q", language="shell", timeout=120.0)
        if result["process_status"] != "completed":
            LOG.warning(f"Failed to upgrade matplotlib: {result.get('stderr', 'Unknown error')}")
        else:
            LOG.info("Successfully upgraded matplotlib")

        # Check if test data exists at /data/test_data.h5
        check_cmd = "test -f /data/test_data.h5 && echo 'exists' || echo 'missing'"
        result, _ = await sandbox.execute_code(check_cmd, language="shell", timeout=10.0)

        if result.get("stdout", "").strip() == "missing":
            LOG.error("Test data not found at /data/test_data.h5")
            raise RuntimeError("Scicode test data not found in sandbox. See logs for details.")
        else:
            LOG.info("Test data found at /data/test_data.h5")

        await sandbox.close()

    asyncio.run(install_packages())

    jsonl_file = eval_config.input_file
    with open(jsonl_file, "rt", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]
    status_lists = test_code(eval_config, data)
    with open(jsonl_file, "wt", encoding="utf-8") as fout:
        for idx, elem in enumerate(data):
            elem["eval_status"] = status_lists[idx]
            fout.write(json.dumps(elem) + "\n")
