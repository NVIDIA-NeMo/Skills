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

import json

# running most things through subprocess since that's how it's usually used
import subprocess
from unittest.mock import MagicMock

import pytest

from nemo_skills.evaluation.metrics import ComputeMetrics
from nemo_skills.pipeline.generate import _create_commandgroup_from_config


def test_eval_gsm8k_api(tmp_path):
    cmd = (
        f"ns eval "
        f"    --server_type=openai "
        f"    --model=nvidia/nvidia-nemotron-nano-9b-v2 "
        f"    --server_address=https://integrate.api.nvidia.com/v1 "
        f"    --benchmarks=gsm8k "
        f"    --output_dir={tmp_path} "
        f"    ++max_samples=2 "
    )
    subprocess.run(cmd, shell=True, check=True)

    # checking that summarize results works (just that there are no errors, but can inspect the output as well)
    subprocess.run(
        f"ns summarize_results {tmp_path}",
        shell=True,
        check=True,
    )

    # running compute_metrics to check that results are expected
    metrics = ComputeMetrics(benchmark="gsm8k").compute_metrics(
        [f"{tmp_path}/eval-results/gsm8k/output.jsonl"],
    )["_all_"]["pass@1"]

    assert metrics["symbolic_correct"] >= 80


def test_eval_judge_api(tmp_path):
    cmd = (
        f"ns eval "
        f"    --server_type=openai "
        f"    --model=nvidia/nvidia-nemotron-nano-9b-v2 "
        f"    --server_address=https://integrate.api.nvidia.com/v1 "
        f"    --benchmarks=math-500 "
        f"    --output_dir={tmp_path} "
        f"    --judge_model=nvidia/nvidia-nemotron-nano-9b-v2 "
        f"    --judge_server_address=https://integrate.api.nvidia.com/v1 "
        f"    --judge_server_type=openai "
        f"    --judge_generation_type=math_judge "
        f"    ++max_samples=2 "
    )
    subprocess.run(cmd, shell=True, check=True)

    # checking that summarize results works (just that there are no errors, but can inspect the output as well)
    subprocess.run(
        f"ns summarize_results {tmp_path}",
        shell=True,
        check=True,
    )

    # running compute_metrics to check that results are expected
    metrics = ComputeMetrics(benchmark="math-500").compute_metrics(
        [f"{tmp_path}/eval-results/math-500/output.jsonl"],
    )["_all_"]["pass@1"]

    assert metrics["symbolic_correct"] >= 40
    assert metrics["judge_correct"] >= 40


def test_fail_on_api_key_env_var(tmp_path):
    cmd = (
        f"ns eval "
        f"    --server_type=openai "
        f"    --model=nvidia/nvidia-nemotron-nano-9b-v2 "
        f"    --server_address=https://integrate.api.nvidia.com/v1 "
        f"    --benchmarks=gsm8k "
        f"    --output_dir={tmp_path} "
        f"    ++max_samples=2 "
        f"    ++server.api_key_env_var=MY_CUSTOM_KEY "
    )
    result = subprocess.run(cmd, shell=True, check=True, capture_output=True)

    # nemo-run always finishes with 0 error code, so just checking that expected exception is in the output
    assert (
        "ValueError: You defined api_key_env_var=MY_CUSTOM_KEY but the value is not set" in result.stdout.decode()
    ), result.stdout.decode()


def test_succeed_on_api_key_env_var(tmp_path):
    cmd = (
        f"export MY_CUSTOM_KEY=$NVIDIA_API_KEY && "
        f"unset NVIDIA_API_KEY && "
        f"ns eval "
        f"    --server_type=openai "
        f"    --model=nvidia/nvidia-nemotron-nano-9b-v2 "
        f"    --server_address=https://integrate.api.nvidia.com/v1 "
        f"    --benchmarks=gsm8k "
        f"    --output_dir={tmp_path} "
        f"    ++max_samples=2 "
        f"    ++server.api_key_env_var=MY_CUSTOM_KEY "
    )
    subprocess.run(cmd, shell=True, check=True)

    # checking that summarize results works (just that there are no errors, but can inspect the output as well)
    subprocess.run(
        f"ns summarize_results {tmp_path}",
        shell=True,
        check=True,
    )

    # running compute_metrics to check that results are expected
    metrics = ComputeMetrics(benchmark="gsm8k").compute_metrics(
        [f"{tmp_path}/eval-results/gsm8k/output.jsonl"],
    )["_all_"]["pass@1"]

    assert metrics["symbolic_correct"] >= 80


@pytest.mark.parametrize("format", ["list", "dict"])
def test_generate_openai_format(tmp_path, format):
    cmd = (
        f"ns generate "
        f"    --server_type=openai "
        f"    --model=nvidia/nvidia-nemotron-nano-9b-v2 "
        f"    --server_address=https://integrate.api.nvidia.com/v1 "
        f"    --input_file=/nemo_run/code/tests/data/openai-input-{format}.test "
        f"    --output_dir={tmp_path} "
        f"    ++prompt_format=openai "
    )
    subprocess.run(cmd, shell=True, check=True)

    # checking that output exists and has the expected format
    with open(f"{tmp_path}/output.jsonl") as fin:
        data = [json.loads(line) for line in fin.readlines()]
    assert len(data) == 2
    assert len(data[0]["generation"]) > 0
    assert len(data[1]["generation"]) > 0


def test_server_metadata_from_num_tasks():
    """Test that metadata dict is properly created from server command returning (cmd, num_tasks)."""
    mock_server_fn = MagicMock(return_value=("python server.py", 4))
    cluster_config = {
        "containers": {"vllm": "nvcr.io/nvidia/nemo:vllm", "nemo-skills": "nvcr.io/nvidia/nemo:skills"},
        "executor": "slurm",
    }
    server_config = {
        "server_type": "vllm",
        "num_gpus": 8,
        "num_nodes": 1,
        "model_path": "/models/test",
        "server_port": 5000,
    }

    cmd_group = _create_commandgroup_from_config(
        generation_cmd="python generate.py",
        server_config=server_config,
        with_sandbox=False,
        sandbox_port=None,
        cluster_config=cluster_config,
        installation_command=None,
        get_server_command_fn=mock_server_fn,
        partition=None,
        keep_mounts_for_sandbox=False,
        task_name="test-task",
        log_dir="/tmp/logs",
    )

    server_cmd = cmd_group.commands[0]
    assert isinstance(server_cmd.metadata, dict)
    assert server_cmd.metadata["num_tasks"] == 4
    assert server_cmd.metadata["gpus"] == 8
