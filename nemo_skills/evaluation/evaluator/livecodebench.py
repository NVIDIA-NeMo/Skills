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

import asyncio
import json
import logging
import shlex
import shutil
import subprocess
import sys
import textwrap
from contextlib import asynccontextmanager
from dataclasses import field
from typing import Any, Dict, List, Tuple

import httpx

from nemo_skills.code_execution.sandbox import Sandbox, get_sandbox
from nemo_skills.evaluation.evaluator.base import BaseEvaluatorConfig
from nemo_skills.evaluation.evaluator.code import preprocess_code
from nemo_skills.utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))

LIVECODEBENCH_GIT_URL = "git+https://github.com/wasiahmad/livecodebench.git"


@nested_dataclass(kw_only=True)
class LiveCodeBenchEvaluatorConfig(BaseEvaluatorConfig):
    sandbox: dict = field(default_factory=lambda: {"sandbox_type": "local"})
    language: str = "python"  # use either "python" or "cpp"
    test_file: str = None
    interpreter: str = "python"  # use either "python" or "pypy3"
    timeout: int = 6
    num_processes: int = 12
    timeout_buffer: int = 60
    num_retries: int = 3


@asynccontextmanager
async def sandbox_context(config: dict):
    """Provides a managed sandbox instance."""
    sandbox = get_sandbox(**config)
    try:
        yield sandbox
    finally:
        LOG.info("Closing sandbox...")
        await sandbox.close()


async def execute_in_sandbox_with_retries(
    sandbox: Sandbox,
    num_retries: int,
    *args,
    **kwargs,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Executes a command in the sandbox, retrying on network errors.

    Uses exponential backoff for delays between retries.
    """
    last_exception = None
    for attempt in range(num_retries):
        try:
            return await sandbox.execute_code(*args, **kwargs)
        except httpx.NetworkError as e:
            last_exception = e
            if attempt == num_retries - 1:
                break

            wait_time = 2**attempt
            LOG.warning(
                f"Sandbox connection error (attempt {attempt + 1}/{num_retries}): {e}. "
                f"Retrying in {wait_time} second(s)..."
            )
            await asyncio.sleep(wait_time)

    LOG.error(f"All {num_retries} sandbox connection attempts failed.")
    raise last_exception


async def is_sandbox_available(sandbox_config: dict) -> bool:
    """
    Checks if the sandbox service is running and accessible by sending a test request.

    Args:
        sandbox_config: The configuration dictionary for the sandbox.

    Returns:
        True if a connection can be established, False otherwise.
    """
    LOG.info(f"Attempting to connect to sandbox with config: {sandbox_config}")
    try:
        async with sandbox_context(sandbox_config) as sandbox:
            await execute_in_sandbox_with_retries(sandbox, 1, "true", language="shell", timeout=5)
        LOG.info("Sandbox connection successful. Sandbox is available.")
        return True
    except httpx.NetworkError as e:
        LOG.warning(f"Sandbox is unavailable due to a network error: {type(e).__name__} - {e}")
        return False
    except Exception as e:
        LOG.warning(f"An unexpected error occurred while checking sandbox availability: {e}")
        return False


def _preprocess_and_validate_file(jsonl_file: str, language: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Reads a JSONL file, preprocesses samples, and validates the release version.
    This function consolidates logic previously duplicated in both evaluation functions.
    """
    with open(jsonl_file, "r", encoding="utf-8") as f_in:
        samples = [preprocess_code(json.loads(line), language) for line in f_in]

    if not samples:
        raise ValueError(f"No samples found in {jsonl_file}")

    # Validate that all samples share the same release version
    versions = {s["release_version"] for s in samples}
    if len(versions) > 1:
        raise ValueError(f"All samples should have the same release version. Found: {versions}")
    release_version = versions.pop()

    # Prepare samples for evaluation
    for s in samples:
        s["question_id"] = s["task_id"]  # Required by the evaluator
        s["code_list"] = [s["completion"]]

    # Overwrite the file with the processed samples
    with open(jsonl_file, "w", encoding="utf-8") as f_out:
        f_out.writelines(json.dumps(sample) + "\n" for sample in samples)

    return samples, release_version


def _postprocess_results(jsonl_file: str, samples: List[Dict[str, Any]]):
    """
    Reads evaluation results, merges them into the samples, and saves the final output.
    This function consolidates logic previously duplicated in both evaluation functions.
    """
    results_file = jsonl_file[:-6] + "_eval_results.json"
    with open(results_file, "r", encoding="utf-8") as f_in:
        eval_grades = json.load(f_in)

    with open(jsonl_file, "w", encoding="utf-8") as f_out:
        for s in samples:
            s["graded_list"] = eval_grades["eval"][s["task_id"]]["graded_list"]
            f_out.write(json.dumps(s) + "\n")

    # Move eval file to ensure metrics are recomputed if run again
    shutil.move(results_file, jsonl_file[:-6] + "_eval_results-saved.json")
    LOG.info(f"Finished processing {jsonl_file}, results saved.")


async def _install_packages_in_sandbox(sandbox: Sandbox, eval_config: LiveCodeBenchEvaluatorConfig) -> bool:
    """Installs required packages in the provided sandbox."""
    LOG.info(f"Installing livecodebench with {eval_config.interpreter} in sandbox...")
    pip_cmd = "pip" if eval_config.interpreter == "python" else "pypy3 -m pip"
    cmd = f"{pip_cmd} install {LIVECODEBENCH_GIT_URL}"

    result, _ = await execute_in_sandbox_with_retries(
        sandbox, eval_config.num_retries, cmd, language="shell", timeout=300
    )
    if result.get("process_status") != "completed":
        LOG.warning(f"Failed to install livecodebench: {result.get('stderr', 'Unknown error')}")
        return False

    LOG.info("Successfully installed livecodebench.")
    return True


def _install_packages_locally(interpreter: str):
    """Installs packages on the local machine."""
    try:
        from livecodebench.evaluate import evaluate

        return evaluate  # Return the function if already installed
    except ImportError:
        LOG.info("Package 'livecodebench' not found. Attempting to install...")
        try:
            # Use the specified python interpreter for installation
            pip_executable = sys.executable if interpreter == "python" else interpreter
            subprocess.check_call([pip_executable, "-m", "pip", "install", LIVECODEBENCH_GIT_URL])
            LOG.info("Package installed successfully!")
            from livecodebench.evaluate import evaluate

            return evaluate
        except (subprocess.CalledProcessError, ImportError) as e:
            LOG.error(f"Failed to install/import 'livecodebench'. Please install it manually. Error: {e}")
            raise


async def eval_livecodebench_async(eval_config: LiveCodeBenchEvaluatorConfig):
    """Evaluation running within a sandbox."""
    async with sandbox_context(eval_config.sandbox) as sandbox:
        if not await _install_packages_in_sandbox(sandbox, eval_config):
            return

        jsonl_file = eval_config.input_file
        LOG.info(f"Processing file: {jsonl_file} in sandbox")
        try:
            samples, release_version = _preprocess_and_validate_file(jsonl_file, eval_config.language)
        except ValueError as e:
            LOG.error(f"Skipping {jsonl_file} due to pre-processing error: {e}")

        if eval_config.language == "python":
            release_version = f"release_{release_version}"
        test_file_arg = repr(eval_config.test_file) if eval_config.test_file else "None"
        eval_code = textwrap.dedent(f"""
            from livecodebench.evaluate import evaluate
            evaluate(
                custom_output_file='{jsonl_file}',
                release_version='{release_version}',
                test_file={test_file_arg},
                k_list=[1],
                language='{eval_config.language}',
                num_process_evaluate={eval_config.num_processes},
                timeout={eval_config.timeout}
            )
        """)

        cmd = f"{eval_config.interpreter} -c {shlex.quote(eval_code)}"
        output, _ = await execute_in_sandbox_with_retries(
            sandbox,
            eval_config.num_retries,
            cmd,
            language="shell",
            timeout=eval_config.timeout * len(samples) + eval_config.timeout_buffer,
            max_output_characters=100_000,
        )

        if output.get("process_status") != "completed":
            LOG.error(f"Evaluation failed for {jsonl_file}. Stderr: {output.get('stderr')}")

        _postprocess_results(jsonl_file, samples)


def eval_livecodebench_without_sandbox(eval_config: LiveCodeBenchEvaluatorConfig):
    """Evaluation running on the local machine."""
    evaluate_fn = _install_packages_locally(eval_config.interpreter)
    if not evaluate_fn:
        return

    jsonl_file = eval_config.input_file
    LOG.info(f"Processing file: {jsonl_file} locally")
    try:
        samples, release_version = _preprocess_and_validate_file(jsonl_file, eval_config.language)
    except ValueError as e:
        LOG.error(f"Skipping {jsonl_file} due to pre-processing error: {e}")

    if eval_config.language == "python":
        release_version = f"release_{release_version}"

    evaluate_fn(
        custom_output_file=jsonl_file,
        release_version=release_version,
        k_list=[1],
        language=eval_config.language,
        test_file=eval_config.test_file,
        num_process_evaluate=eval_config.num_processes,
        timeout=eval_config.timeout,
    )

    _postprocess_results(jsonl_file, samples)


def eval_livecodebench(cfg):
    """Main entry point for LiveCodeBench evaluation."""
    eval_config = LiveCodeBenchEvaluatorConfig(_init_nested=True, **cfg)

    if eval_config.language == "python" and eval_config.interpreter not in ["python", "pypy3"]:
        raise ValueError("Python interpreter must be 'python' or 'pypy3'.")
    if eval_config.language == "cpp" and eval_config.interpreter != "python":
        LOG.warning(
            f"For C++ evaluation, the harness must be run with 'python'. "
            f"Ignoring configured interpreter '{eval_config.interpreter}' and using 'python'."
        )
        eval_config.interpreter = "python"

    sandbox_is_ready = asyncio.run(is_sandbox_available(eval_config.sandbox))
    if eval_config.interpreter == "pypy3" and not sandbox_is_ready:
        raise RuntimeError("The 'pypy3' interpreter requires a running sandbox, but the service was unreachable.")

    if sandbox_is_ready:
        asyncio.run(eval_livecodebench_async(eval_config))
    else:
        eval_livecodebench_without_sandbox(eval_config)
