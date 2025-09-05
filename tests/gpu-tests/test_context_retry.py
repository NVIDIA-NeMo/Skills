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

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pytest

from tests.conftest import docker_rm, docker_rm_and_mkdir, docker_run


@dataclass
class TestConfig:
    """Configuration for test parameters"""

    num_tokens_to_generate: int = 1_000_000
    num_samples: int = 4
    accuracy_threshold_percent: int = 25
    server_gpus: int = 1
    server_nodes: int = 1


@dataclass
class TestEnvironment:
    """Manages test environment setup and validation"""

    model_path = os.getenv("NEMO_SKILLS_TEST_HF_MODEL")
    model_type = os.getenv("NEMO_SKILLS_TEST_MODEL_TYPE")

    def validate_environment(self):
        """Validate required environment variables"""
        if not self.model_path:
            pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test", allow_module_level=True)
        if not self.model_type:
            pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test", allow_module_level=True)


class CommandBuilder:
    """Builds command strings for different test scenarios"""

    def __init__(self, env: TestEnvironment, config: TestConfig):
        self.env = env
        self.config = config
        self.base_path = Path(__file__).absolute().parent

    def _build_base_cmd(self, cmd_type: str, output_dir: str, server_type: str) -> str:
        """Build base command with common parameters"""
        return (
            f"ns {cmd_type} "
            f"    --cluster test-local --config_dir {self.base_path} "
            f"    --model {self.env.model_path} "
            f"    --server_type {server_type} "
            f"    --output_dir {output_dir} "
            f"    --server_gpus {self.config.server_gpus} "
            f"    --server_nodes {self.config.server_nodes} "
        )

    def build_eval_cmd(self, output_dir: str, server_type: str, enable_soft_fail: bool, retry_strategy: str) -> str:
        """Build evaluation command"""
        base = self._build_base_cmd("eval", output_dir, server_type)
        return base + (
            f"    --benchmarks gsm8k "
            f"    ++max_samples={self.config.num_samples} "
            f"    ++inference.tokens_to_generate={self.config.num_tokens_to_generate} "
            f"    ++server.enable_soft_fail={enable_soft_fail} "
            + (f"    ++server.context_limit_retry_strategy={retry_strategy} " if retry_strategy else "")
        )

    def build_generate_cmd(self, output_dir: str, server_type: str, input_file: str, retry_strategy: str) -> str:
        """Build generation command"""
        base = self._build_base_cmd("generate", output_dir, server_type)
        return base + (
            f"    --input_file {input_file} "
            f"    ++prompt_config=generic/default "
            f"    ++server.enable_soft_fail=True "
            f"    ++server.context_limit_retry_strategy={retry_strategy} "
        )


class OutputManager:
    """Manages test output directories and cleanup"""

    @staticmethod
    def setup_output_dir(model_type: str, test_name: str) -> str:
        """Setup and clean output directory"""
        output_dir = f"/tmp/nemo-skills-tests/{model_type}/{test_name}"
        docker_rm([output_dir])
        return output_dir

    @staticmethod
    def setup_io_files(output_dir: str) -> tuple[str, str]:
        """Setup input and output files for generation tests"""
        input_file = f"{output_dir}/input.jsonl"
        output_file = f"{output_dir}/output.jsonl"
        docker_rm_and_mkdir(input_file)
        docker_rm_and_mkdir(output_file)
        return input_file, output_file


class MetricsValidator:
    """Validates test results and metrics"""

    def __init__(self, config: TestConfig):
        self.config = config

    def validate_eval_metrics(self, output_dir: str) -> Dict[str, Any]:
        """Validate evaluation metrics from results file"""
        metrics_file = f"{output_dir}/eval-results/gsm8k/metrics.json"

        with open(metrics_file, "r") as f:
            metrics = json.load(f)["gsm8k"]["pass@1"]

        # Validate basic metrics
        assert metrics["num_entries"] == self.config.num_samples

        # Model-specific validation (can be extended)
        accuracy_threshold = self.config.accuracy_threshold_percent
        assert metrics["symbolic_correct"] >= accuracy_threshold

        return metrics

    def validate_eval_failure(self, output_dir: str) -> bool:
        """Validate that evaluation failed as expected"""
        metrics_file = f"{output_dir}/eval-results/gsm8k/metrics.json"

        try:
            with open(metrics_file, "r") as f:
                metrics = json.load(f)["gsm8k"]["pass@1"]
            return metrics["num_entries"] != self.config.num_samples
        except FileNotFoundError:
            return True  # No metrics file indicates failure

    def validate_eval_completion_but_empty_generation(self, output_dir: str) -> bool:
        """Validate that evaluation completion is successful"""
        metrics_file = f"{output_dir}/eval-results/gsm8k/metrics.json"
        assert os.path.exists(metrics_file), "Metrics file not found"

        with open(metrics_file, "r") as f:
            metrics = json.load(f)["gsm8k"]["pass@1"]

        # The generation is empty for all samples, so the metrics should be 0
        assert metrics["num_entries"] == self.config.num_samples
        assert metrics["symbolic_correct"] == 0

        return True

    def validate_generation_output(self, output_file: str) -> bool:
        """Validate that generation output exists"""
        return os.path.exists(output_file)


def _create_large_input_file(input_file: str, num_samples: int):
    """Create a fake input jsonl file with long prompts"""
    # TODO: Currently this is just a single turn message. Need to add tests for multi-turn messages.
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        try:
            for _ in range(num_samples):
                # Create large prompt that will likely exceed context limits
                large_prompt = {"question": "a" * 500_000 + "b" * 500_000}
                temp_file.write(json.dumps(large_prompt).encode())
                temp_file.write(b"\n")
            temp_file.flush()

            # Copy to Docker container
            docker_run(f"cp {temp_file.name} {input_file}")
        finally:
            # Clean up temporary file
            os.unlink(temp_file.name)


class ContextRetryTestSuite:
    """Main test suite for context retry functionality"""

    def __init__(self):
        self.env = TestEnvironment()
        self.env.validate_environment()  # validate environment before initializing other objects
        self.config = TestConfig()
        self.cmd_builder = CommandBuilder(self.env, self.config)
        self.output_manager = OutputManager()
        self.validator = MetricsValidator(self.config)

    def run_no_strategy_test(self, server_type: str, test_name: str, enable_soft_fail: bool, retry_strategy: str):
        """Run evaluation test for completion."""
        output_dir = self.output_manager.setup_output_dir(self.env.model_type, test_name)
        cmd = self.cmd_builder.build_eval_cmd(output_dir, server_type, enable_soft_fail, retry_strategy)
        subprocess.run(cmd, shell=True, check=True)

        self.validator.validate_eval_completion_but_empty_generation(output_dir)

    def run_reduce_generation_test(
        self,
        server_type: str,
        test_name: str,
        enable_soft_fail: bool,
        retry_strategy: str,
        expect_success: bool = True,
    ):
        """Run evaluation test with specified parameters"""
        output_dir = self.output_manager.setup_output_dir(self.env.model_type, test_name)
        cmd = self.cmd_builder.build_eval_cmd(output_dir, server_type, enable_soft_fail, retry_strategy)

        subprocess.run(cmd, shell=True, check=True)

        if expect_success:
            return self.validator.validate_eval_metrics(output_dir)
        else:
            return self.validator.validate_eval_failure(output_dir)

    def run_reduce_prompt_test(self, server_type: str, test_name: str, retry_strategy: str, extra_args: str = ""):
        """Run generation test with specified retry strategy using reducing the prompt"""
        output_dir = self.output_manager.setup_output_dir(self.env.model_type, test_name)
        input_file, output_file = self.output_manager.setup_io_files(output_dir)

        _create_large_input_file(input_file, num_samples=1)

        cmd = self.cmd_builder.build_generate_cmd(output_dir, server_type, input_file, retry_strategy) + extra_args
        subprocess.run(cmd, shell=True, check=True)

        assert self.validator.validate_generation_output(output_file), "Output file not found"


# Initialize test suite
test_suite = ContextRetryTestSuite()


@pytest.mark.gpu
@pytest.mark.parametrize("server_type", ["trtllm", "sglang", "vllm"])
def test_context_retry_no_strategy(server_type):
    """Test that the generation finishes successfully if soft fail is enabled and the strategy is reduce_generation."""
    test_suite.run_no_strategy_test(
        server_type=server_type,
        test_name=f"{server_type}-eval-no-strategy",
        enable_soft_fail=True,
        retry_strategy=None,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("server_type", ["sglang", "vllm"])
def test_context_retry_reduce_generation_enabled(server_type):
    """Test that the generation finishes successfully if soft fail is enabled and the strategy is reduce_generation."""
    test_suite.run_reduce_generation_test(
        server_type=server_type,
        test_name=f"{server_type}-eval-reduce-generation-enabled",
        enable_soft_fail=True,
        retry_strategy="reduce_generation",
    )


@pytest.mark.gpu
@pytest.mark.parametrize("server_type", ["trtllm", "sglang", "vllm"])
def test_context_retry_disabled(server_type):
    """Test that the generation doesn't finish successfully if soft fail is disabled."""
    result = test_suite.run_reduce_generation_test(
        server_type=server_type,
        test_name=f"{server_type}-eval-reduce-generation-disabled",
        enable_soft_fail=False,
        retry_strategy=None,
        expect_success=False,
    )
    assert result, "Expected test to fail but it succeeded"


@pytest.mark.gpu
@pytest.mark.parametrize("server_type", ["sglang", "vllm"])
def test_context_retry_reduce_prompt_start(server_type):
    # TODO: Currently this is just a single turn message. Need to add tests for multi-turn messages.
    """Test that successful generation is possible if soft fail is enabled and the strategy is reduce_prompt, removing tokens from the start."""

    extra_args = (
        " ++inference.tokens_to_generate=2048 "  # Setting this otherwise the prompt reduction strategy will fail
    )
    test_suite.run_reduce_prompt_test(
        server_type=server_type,
        test_name=f"{server_type}-eval-reduce-prompt-start",
        retry_strategy="reduce_prompt_from_start",
        extra_args=extra_args,
    )


@pytest.mark.gpu
@pytest.mark.parametrize("server_type", ["sglang", "vllm"])
def test_context_retry_reduce_prompt_end(server_type):
    # TODO: Currently this is just a single turn message. Need to add tests for multi-turn messages.
    """Test that successful generation is possible if soft fail is enabled and the strategy is reduce_prompt, removing tokens from the end."""
    extra_args = (
        " ++inference.tokens_to_generate=2048 "  # Setting this otherwise the prompt reduction strategy will fail
    )
    test_suite.run_reduce_prompt_test(
        server_type=server_type,
        test_name=f"{server_type}-eval-reduce-prompt-end",
        retry_strategy="reduce_prompt_from_end",
        extra_args=extra_args,
    )
