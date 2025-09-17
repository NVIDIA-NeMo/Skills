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

import pytest

from nemo_skills.evaluation.metrics import ComputeMetrics


def test_eval_gsm8k_api(tmp_path):
    cmd = (
        f"ns eval "
        f"    --server_type=openai "
        f"    --model=meta/llama-3.1-8b-instruct "
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
        f"    --model=meta/llama-3.1-8b-instruct "
        f"    --server_address=https://integrate.api.nvidia.com/v1 "
        f"    --benchmarks=math-500 "
        f"    --output_dir={tmp_path} "
        f"    --judge_model=meta/llama-3.1-8b-instruct "
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
        f"    --model=meta/llama-3.1-8b-instruct "
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
        f"    --model=meta/llama-3.1-8b-instruct "
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
        f"    --model=meta/llama-3.1-8b-instruct "
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
