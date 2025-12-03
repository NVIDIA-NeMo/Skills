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
from pathlib import Path

import pytest
from utils import require_env_var

from tests.conftest import docker_rm

# NOTE: Tool calling behavior is model-specific. Some models (e.g., Qwen) work with standard
# tool call parsers without requiring `tool_choice: "auto"` in the request body, while others
# (e.g., Kimi-K2) have non-standard tool_call_id formats that may require custom handling.
# See: https://huggingface.co/moonshotai/Kimi-K2-Instruct/discussions/48
# For models that require `tool_choice: "auto"`, use a custom model class via the `model_class`
# parameter (e.g., ++server.model_class=nemo_skills.inference.model.sglang::SGLangModel).

# Test prompts designed to strongly encourage tool use
TEST_PROMPTS = [
    {"problem": "Use the python tool to calculate 7 * 8 + 15 and verify your result is correct."},
    {"problem": "Use python to compute the factorial of 10 and verify your answer."},
    {"problem": "Write python code to check if 17 is a prime number and confirm the result."},
]


def _create_test_input_file():
    """Create a temporary input file with test prompts."""
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as f:
        for prompt in TEST_PROMPTS:
            f.write(json.dumps(prompt) + "\n")
    return path


def _run_tool_calling_test(server_type: str, server_args: str, output_dir: str):
    """Common test logic for tool calling with different server types."""
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")

    docker_rm([output_dir])

    # Create test input file
    input_file = _create_test_input_file()

    try:
        cmd = (
            f"ns generate "
            f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
            f"    --model {model_path} "
            f"    --output_dir {output_dir} "
            f"    --server_type {server_type} "
            f"    --server_gpus 1 "
            f"    --server_nodes 1 "
            f"    --server_args '{server_args}' "
            f"    --with_sandbox "
            f"    --input_file {input_file} "
            f"    ++tool_modules=[nemo_skills.mcp.servers.python_tool::PythonTool] "
            f"    ++prompt_config=generic/math "
            f"    ++inference.tokens_to_generate=4096 "
            f"    ++inference.temperature=0.6 "
            f"    ++skip_filled=False "
        )
        subprocess.run(cmd, shell=True, check=True)

        # Verify output exists and tool calls were made
        output_file = f"{output_dir}/output.jsonl"
        print(f"\n=== Output file location: {output_file} ===")
        assert os.path.exists(output_file), f"Output file not found: {output_file}"
        assert os.path.exists(f"{output_file}.done"), "Done marker not found"

        with open(output_file) as fin:
            lines = fin.readlines()

        assert len(lines) == len(TEST_PROMPTS), f"Expected {len(TEST_PROMPTS)} lines, got {len(lines)}"

        # Check that tool calls were made for each sample
        samples_with_tool_calls = 0
        for line in lines:
            data = json.loads(line)
            assert "generation" in data, "Missing 'generation' field in output"
            num_tool_calls = data.get("num_tool_calls", 0)
            if num_tool_calls > 0:
                samples_with_tool_calls += 1

        # At least some samples should have made tool calls
        assert samples_with_tool_calls > 0, (
            "No samples made tool calls. Expected tool usage for prompts that explicitly request it."
        )

    finally:
        # Clean up temp file
        if os.path.exists(input_file):
            os.remove(input_file)


@pytest.mark.gpu
def test_vllm_tool_calling():
    """Test that VLLM properly makes tool calls with --enable-auto-tool-choice."""
    model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")
    output_dir = f"/tmp/nemo-skills-tests/{model_type}/vllm-tool-calling/generation"

    _run_tool_calling_test(
        server_type="vllm",
        server_args="--enforce-eager --max-model-len 8192 --enable-auto-tool-choice --tool-call-parser hermes",
        output_dir=output_dir,
    )


@pytest.mark.gpu
def test_sglang_tool_calling():
    """Test that SGLang properly makes tool calls with tool_choice='auto' in request body."""
    model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")
    output_dir = f"/tmp/nemo-skills-tests/{model_type}/sglang-tool-calling/generation"

    _run_tool_calling_test(
        server_type="sglang",
        server_args="--context-length 8192 --tool-call-parser qwen25",
        output_dir=output_dir,
    )
