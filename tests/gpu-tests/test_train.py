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

import os
from pathlib import Path

import pytest
from utils import require_env_var

from nemo_skills.pipeline.cli import grpo_nemo_rl, sft_nemo_rl, wrap_arguments
from tests.conftest import docker_rm


def _copy_test_data_to_tmp(filename: str, model_type: str) -> str:
    test_data_dir = Path(__file__).absolute().parents[1] / "data"
    output_data_dir = Path("/tmp") / f"nemo-skills-training-data-{os.getuid()}" / model_type
    output_data_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_data_dir / f"{Path(filename).stem}.jsonl"
    output_file.write_text((test_data_dir / filename).read_text())
    return str(output_file)


def _assert_final_hf_model_exists(output_dir: str) -> None:
    assert (Path(output_dir) / "final_hf_model" / "config.json").exists()


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["fsdp", "megatron"])
def test_sft_nemo_rl(backend):
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")
    model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/test-sft-nemo-rl/{backend}"
    training_data = _copy_test_data_to_tmp("small-sft-data.test", model_type)

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
        training_data=training_data,
        disable_wandb=True,
    )
    _assert_final_hf_model_exists(output_dir)


@pytest.mark.gpu
def test_sft_nemo_rl_messages_format():
    """Test SFT training with messages format data."""
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")
    model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/test-sft-nemo-rl-messages/megatron"
    training_data = _copy_test_data_to_tmp("small-sft-data-messages.test", model_type)

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
        training_data=training_data,
        disable_wandb=True,
    )
    _assert_final_hf_model_exists(output_dir)


@pytest.mark.gpu
@pytest.mark.parametrize("backend", ["fsdp", "megatron"])
def test_grpo_nemo_rl(backend):
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")
    model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/test-grpo-nemo-rl/{backend}"
    training_data = _copy_test_data_to_tmp("small-nemo-gym-grpo-data.test", model_type)

    # need to clean up current cluster configuration as we mount /tmp and it causes problems
    # need to clean up cache folder as otherwise megatron backend might fail when checkpoint format changes
    docker_rm(["/tmp/ray/ray_current_cluster", "/mnt/datadrive/nemo-skills-test-data/hf-cache/nemo_rl/", output_dir])

    common_overrides = (
        "++grpo.max_num_steps=1 "
        "++grpo.val_period=0 "
        "++grpo.val_at_start=False "
        "++grpo.val_at_end=False "
        "++grpo.num_prompts_per_step=1 "
        "++grpo.num_generations_per_prompt=2 "
        "++grpo.num_val_generations_per_prompt=1 "
        "++grpo.overlong_filtering=False "
        "++grpo.use_leave_one_out_baseline=False "
        "++checkpointing.save_period=1 "
        "++checkpointing.save_optimizer=False "
        "++policy.train_global_batch_size=2 "
        "++policy.train_micro_batch_size=1 "
        "++policy.logprob_batch_size=1 "
        "++policy.max_total_sequence_length=256 "
        "++policy.make_sequence_length_divisible_by=1 "
        "++policy.sequence_packing.enabled=False "
        "++policy.dynamic_batching.enabled=False "
        "++policy.generation.max_new_tokens=64 "
        "++policy.generation.vllm_cfg.tensor_parallel_size=1 "
        "++policy.generation.vllm_cfg.max_model_len=256 "
        "++policy.generation.vllm_cfg.gpu_memory_utilization=0.3 "
        "++policy.generation.vllm_cfg.http_server_serving_chat_kwargs.enable_auto_tools=False "
        "++policy.generation.vllm_cfg.http_server_serving_chat_kwargs.tool_parser=null "
        "++policy.generation.vllm_cfg.http_server_serving_chat_kwargs.reasoning_parser=null "
        "++env.nemo_gym.config_paths=[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,resources_servers/math_with_judge/configs/math_with_judge.yaml] "
    )
    backend_overrides = {
        "fsdp": (
            "++policy.dtensor_cfg.tensor_parallel_size=1 "
            "++policy.dtensor_cfg.context_parallel_size=1 "
            "++policy.dtensor_cfg.sequence_parallel=False "
            "++policy.dtensor_cfg.activation_checkpointing=False "
            "++policy.optimizer.name=torch.optim.AdamW "
            "++policy.optimizer.kwargs.lr=1e-6 "
            "++policy.optimizer.kwargs.weight_decay=0.0 "
            "++policy.optimizer.kwargs.betas=[0.9,0.999] "
            "++policy.optimizer.kwargs.eps=1e-8 "
            "++policy.scheduler.name=torch.optim.lr_scheduler.ConstantLR "
            "++policy.scheduler.kwargs.factor=1.0 "
            "++policy.scheduler.kwargs.total_iters=10000000000 "
        ),
        "megatron": (
            "++policy.megatron_cfg.tensor_model_parallel_size=1 "
            "++policy.megatron_cfg.pipeline_model_parallel_size=1 "
            "++policy.megatron_cfg.context_parallel_size=1 "
            "++policy.megatron_cfg.expert_model_parallel_size=1 "
            "++policy.megatron_cfg.expert_tensor_parallel_size=1 "
            "++policy.megatron_cfg.sequence_parallel=False "
            "++policy.megatron_cfg.activation_checkpointing=False "
            "++policy.megatron_cfg.freeze_moe_router=False "
            "++policy.megatron_cfg.moe_router_enable_expert_bias=False "
            "++policy.megatron_cfg.track_moe_metrics=False "
            "++policy.megatron_cfg.moe_per_layer_logging=False "
            "++policy.megatron_cfg.optimizer.lr=1e-6 "
            "++policy.megatron_cfg.optimizer.min_lr=1e-6 "
            "++policy.megatron_cfg.scheduler.lr_warmup_iters=0 "
        ),
    }[backend]

    grpo_nemo_rl(
        ctx=wrap_arguments(common_overrides + backend_overrides),
        cluster="test-local",
        config_dir=Path(__file__).absolute().parent,
        expname="test-grpo-nemo-rl",
        output_dir=output_dir,
        hf_model=model_path,
        num_nodes=1,
        num_gpus=1,
        training_data=training_data,
        validation_data=training_data,
        backend=backend,
        disable_wandb=True,
    )
    _assert_final_hf_model_exists(output_dir)
