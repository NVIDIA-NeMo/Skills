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

from pathlib import Path

import pytest
from utils import require_env_var

from nemo_skills.evaluation.metrics import ComputeMetrics
from nemo_skills.pipeline.cli import eval, grpo_nemo_rl, sft_nemo_rl, wrap_arguments
from tests.conftest import docker_rm


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["fsdp", "megatron"])
def test_sft_nemo_rl(backend):
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")
    model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/test-sft-nemo-rl/{backend}"

    # need to clean up current cluster configuration as we mount /tmp and it causes problems
    # need to clean up cache folder as otherwise megatron backend might fail when checkpoint format changes
    docker_rm(["/tmp/ray/ray_current_cluster", "/mnt/datadrive/nemo-skills-test-data/hf-cache/nemo_rl/", output_dir])

    sft_nemo_rl(
        ctx=wrap_arguments(
            "++sft.max_num_steps=5 "
            "++policy.dtensor_cfg.tensor_parallel_size=1 "
            "++checkpointing.save_period=2 "
            "++policy.train_global_batch_size=2 "
            "++policy.train_micro_batch_size=1 "
            "++policy.optimizer.kwargs.lr=1e-6 "
        ),
        cluster="test-local",
        config_dir=Path(__file__).absolute().parent,
        backend=backend,
        expname="test-sft-nemo-rl",
        output_dir=output_dir,
        hf_model=model_path,
        num_nodes=1,
        num_gpus=1,
        training_data="/nemo_run/code/tests/data/small-sft-data.test",
        disable_wandb=True,
    )

    # checking that the final model can be used for evaluation
    eval(
        ctx=wrap_arguments("++max_samples=10 ++inference.tokens_to_generate=10"),
        cluster="test-local",
        config_dir=Path(__file__).absolute().parent,
        model=f"{output_dir}/final_hf_model",
        server_type="vllm",
        output_dir=f"{output_dir}/evaluation",
        benchmarks="gsm8k",
        server_gpus=1,
        server_nodes=1,
        num_jobs=1,
        server_args="--enforce-eager",
    )

    metrics = ComputeMetrics(benchmark="gsm8k").compute_metrics(
        [f"{output_dir}/evaluation/eval-results/gsm8k/output.jsonl"],
    )["_all_"]["pass@1"]
    # only checking the total, since model is tiny
    assert metrics["num_entries"] == 10


@pytest.mark.gpu
def test_sft_nemo_rl_messages_format():
    """Test SFT training with messages format data and infer_from_data chat template."""
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")
    model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/test-sft-nemo-rl-messages/megatron"

    # need to clean up current cluster configuration as we mount /tmp and it causes problems
    # need to clean up cache folder as otherwise megatron backend might fail when checkpoint format changes
    docker_rm(["/tmp/ray/ray_current_cluster", "/mnt/datadrive/nemo-skills-test-data/hf-cache/nemo_rl/", output_dir])

    sft_nemo_rl(
        ctx=wrap_arguments(
            "++sft.max_num_steps=5 "
            "++policy.dtensor_cfg.tensor_parallel_size=1 "
            "++checkpointing.save_period=2 "
            "++policy.train_global_batch_size=2 "
            "++policy.train_micro_batch_size=1 "
            "++policy.optimizer.kwargs.lr=1e-6 "
        ),
        cluster="test-local",
        config_dir=Path(__file__).absolute().parent,
        backend="megatron",
        expname="test-sft-nemo-rl-messages",
        output_dir=output_dir,
        hf_model=model_path,
        num_nodes=1,
        num_gpus=1,
        training_data="/nemo_run/code/tests/data/small-sft-data-messages.test",
        disable_wandb=True,
    )

    # checking that the final model can be used for evaluation
    eval(
        ctx=wrap_arguments("++max_samples=10 ++inference.tokens_to_generate=10"),
        cluster="test-local",
        config_dir=Path(__file__).absolute().parent,
        model=f"{output_dir}/final_hf_model",
        server_type="vllm",
        output_dir=f"{output_dir}/evaluation",
        benchmarks="gsm8k",
        server_gpus=1,
        server_nodes=1,
        num_jobs=1,
        server_args="--enforce-eager",
    )

    metrics = ComputeMetrics(benchmark="gsm8k").compute_metrics(
        [f"{output_dir}/evaluation/eval-results/gsm8k/output.jsonl"],
    )["_all_"]["pass@1"]
    # only checking the total, since model is tiny
    assert metrics["num_entries"] == 10


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["fsdp", "megatron"])
def test_grpo_nemo_rl(backend):
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")
    model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/test-grpo-nemo-rl/{backend}"

    # need to clean up current cluster configuration as we mount /tmp and it causes problems
    # need to clean up cache folder as otherwise megatron backend might fail when checkpoint format changes
    docker_rm(["/tmp/ray/ray_current_cluster", "/mnt/datadrive/nemo-skills-test-data/hf-cache/nemo_rl/", output_dir])

    grpo_nemo_rl(
        ctx=wrap_arguments(
            "++data.prompt.prompt_config=qwen/math-cot "
            "++grpo.max_num_steps=5 "
            "++grpo.num_prompts_per_step=2 "
            "++policy.max_total_sequence_length=256 "
            "++policy.dtensor_cfg.tensor_parallel_size=1 "
            "++checkpointing.save_period=2 "
            "++policy.train_global_batch_size=2 "
            "++policy.train_micro_batch_size=1 "
            "++policy.optimizer.kwargs.lr=1e-6 "
        ),
        cluster="test-local",
        config_dir=Path(__file__).absolute().parent,
        expname="test-grpo-nemo-rl",
        output_dir=output_dir,
        hf_model=model_path,
        num_nodes=1,
        num_gpus=1,
        training_data="/nemo_run/code/tests/data/small-grpo-data.test",
        backend=backend,
        disable_wandb=True,
    )

    # checking that the final model can be used for evaluation
    eval(
        ctx=wrap_arguments("++max_samples=10 ++inference.tokens_to_generate=10"),
        cluster="test-local",
        config_dir=Path(__file__).absolute().parent,
        model=f"{output_dir}/final_hf_model",
        server_type="vllm",
        output_dir=f"{output_dir}/evaluation",
        benchmarks="gsm8k",
        server_gpus=1,
        server_nodes=1,
        num_jobs=1,
        server_args="--enforce-eager",
    )

    metrics = ComputeMetrics(benchmark="gsm8k").compute_metrics(
        [f"{output_dir}/evaluation/eval-results/gsm8k/output.jsonl"],
    )["_all_"]["pass@1"]
    # only checking the total, since model is tiny
    assert metrics["num_entries"] == 10
