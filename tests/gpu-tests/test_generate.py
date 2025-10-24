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
import os
import subprocess
from pathlib import Path

import pytest
from utils import require_env_var

from nemo_skills.evaluation.metrics import ComputeMetrics
from tests.conftest import docker_rm

# TODO: retrieval test


@pytest.mark.gpu
def test_vllm_generate_greedy():
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")
    model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/vllm-generate-greedy/generation"
    docker_rm([output_dir])

    cmd = (
        f"ns generate "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --output_dir {output_dir} "
        f"    --server_type vllm "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    --server_args '--enforce-eager' "
        f"    --input_file=/nemo_run/code/nemo_skills/dataset/gsm8k/test.jsonl "
        f"    ++prompt_config=generic/math "
        f"    ++max_samples=10 "
        f"    ++skip_filled=False "
    )
    subprocess.run(cmd, shell=True, check=True)

    # no evaluation by default - checking just the number of lines and that there is no symbolic_correct key
    with open(f"{output_dir}/output.jsonl") as fin:
        lines = fin.readlines()
    assert len(lines) == 10
    for line in lines:
        data = json.loads(line)
        assert "symbolic_correct" not in data
        assert "generation" in data
    assert os.path.exists(f"{output_dir}/output.jsonl.done")


@pytest.mark.gpu
def test_vllm_generate_greedy_chunked():
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")
    model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/vllm-generate-greedy-chunked/generation"
    docker_rm([output_dir])

    cmd = (
        f"ns generate "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --output_dir {output_dir} "
        f"    --server_type vllm "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    --server_args '--enforce-eager' "
        f"    --num_chunks 2 "
        f"    --input_file=/nemo_run/code/nemo_skills/dataset/gsm8k/test.jsonl "
        f"    ++prompt_config=generic/math "
        f"    ++max_samples=10 "
        f"    ++skip_filled=False "
    )
    subprocess.run(cmd, shell=True, check=True)

    # no evaluation by default - checking just the number of lines and that there is no symbolic_correct key
    with open(f"{output_dir}/output.jsonl") as fin:
        lines = fin.readlines()
    assert len(lines) == 20  # because max_samples is the number of samples per chunk
    for line in lines:
        data = json.loads(line)
        assert "symbolic_correct" not in data
        assert "generation" in data
    assert os.path.exists(f"{output_dir}/output.jsonl.done")


@pytest.mark.gpu
def test_vllm_generate_seeds():
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")
    model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/vllm-generate-seeds/generation"
    docker_rm([output_dir])

    num_seeds = 3
    cmd = (
        f"ns generate "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type vllm "
        f"    --output_dir {output_dir} "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    --server_args '--enforce-eager' "
        f"    --num_random_seeds {num_seeds} "
        f"    --with_sandbox "
        f"    --input_file=/nemo_run/code/nemo_skills/dataset/gsm8k/test.jsonl "
        f"    ++eval_type=math "
        f"    ++prompt_config=generic/math "
        f"    ++max_samples=10 "
        f"    ++skip_filled=False "
    )
    subprocess.run(cmd, shell=True, check=True)

    # checking that all 3 files are created
    for seed in range(num_seeds):
        with open(f"{output_dir}/output-rs{seed}.jsonl") as fin:
            lines = fin.readlines()
        assert len(lines) == 10
        for line in lines:
            data = json.loads(line)
            assert "symbolic_correct" in data
            assert "generation" in data
        assert os.path.exists(f"{output_dir}/output-rs{seed}.jsonl.done")

    # running compute_metrics to check that results are expected
    metrics = ComputeMetrics(benchmark="gsm8k").compute_metrics([f"{output_dir}/output-rs*.jsonl"])["_all_"][
        "majority@3"
    ]
    # rough check, since exact accuracy varies depending on gpu type
    assert metrics["symbolic_correct"] >= 50
    assert metrics["num_entries"] == 10
