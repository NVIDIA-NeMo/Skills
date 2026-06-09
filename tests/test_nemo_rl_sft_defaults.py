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
import re
from pathlib import Path

import yaml

from nemo_skills.pipeline.nemo_rl.sft import NemoRLTask, detect_data_format

TEST_DATA_DIR = Path(__file__).parent / "data"
LEGACY_SFT_CONFIG = TEST_DATA_DIR / "nemo_rl_legacy_sft.yaml"
UPSTREAM_SFT_CONFIG = TEST_DATA_DIR / "nemo_rl_upstream_sft_v0_6.yaml"

PRESERVED_COMMON_DEFAULTS = {
    "checkpointing.keep_top_k",
    "checkpointing.save_period",
    "data.add_bos",
    "data.add_eos",
    "data.num_workers",
    "policy.max_grad_norm",
    "policy.max_total_sequence_length",
    "policy.megatron_cfg.moe_permute_fusion",
    "policy.megatron_cfg.optimizer.adam_eps",
    "policy.megatron_cfg.optimizer.lr",
    "policy.megatron_cfg.optimizer.min_lr",
    "policy.megatron_cfg.optimizer.weight_decay",
    "policy.megatron_cfg.scheduler.lr_decay_iters",
    "policy.megatron_cfg.scheduler.lr_decay_style",
    "policy.megatron_cfg.scheduler.lr_warmup_init",
    "policy.megatron_cfg.scheduler.lr_warmup_iters",
    "policy.optimizer.kwargs.eps",
    "policy.optimizer.kwargs.lr",
    "policy.optimizer.kwargs.weight_decay",
    "policy.sequence_packing.enabled",
    "policy.tokenizer.chat_template",
    "sft.max_num_epochs",
    "sft.max_num_steps",
    "sft.val_at_start",
    "sft.val_batches",
    "sft.val_period",
}

PRESERVED_LEGACY_ONLY_DEFAULTS = {
    "policy.megatron_cfg.layernorm_epsilon",
}

PRESERVED_BY_WRAPPER_DEFAULTS = PRESERVED_COMMON_DEFAULTS | PRESERVED_LEGACY_ONLY_DEFAULTS

# These old top-level aliases were removed upstream. They are now set through
# explicit nested CLI args in recipes/tests when they differ from the default.
CLI_OR_RECIPE_CONTROLLED_COMMON = {
    "policy.dtensor_cfg.context_parallel_size",
    "policy.dtensor_cfg.sequence_parallel",
    "policy.dtensor_cfg.tensor_parallel_size",
    "policy.megatron_cfg.context_parallel_size",
    "policy.megatron_cfg.pipeline_model_parallel_size",
    "policy.megatron_cfg.sequence_parallel",
    "policy.megatron_cfg.tensor_model_parallel_size",
    "policy.model_name",
}

LEGACY_ONLY_SCHEMA_OR_ALIASES = {
    "data.dataset_name",
    "data.force_reprocess",
    "data.input_key",
    "data.output_key",
    "logger.num_val_samples_to_print",
    "policy.context_parallel_size",
    "policy.lr",
    "policy.min_lr",
    "policy.pipeline_model_parallel_size",
    "policy.scheduler",
    "policy.sequence_parallel",
    "policy.tensor_model_parallel_size",
    "policy.weight_decay",
}

# Upstream renamed dataset/logging defaults from `data.dataset_name` to
# `data.train.dataset_name`. The log names are not training behavior and follow
# the upstream schema.
SCHEMA_OR_NON_TRAINING_BEHAVIOR_COMMON = {
    "logger.mlflow.run_name",
    "logger.swanlab.name",
    "logger.tensorboard.log_dir",
    "logger.wandb.name",
}

UPSTREAM_ONLY_SCHEMA_OR_NEMO_RL_OWNED = {
    "checkpointing.save_optimizer",
    "data.default.processor",
    "data.default.prompt_file",
    "data.default.system_prompt_file",
    "data.train.dataset_name",
    "data.train.split",
    "data.validation.dataset_name",
    "data.validation.split",
    "logger.mlflow.tracking_uri",
    "policy.megatron_cfg.force_reconvert_from_hf",
    "policy.megatron_cfg.fp8_cfg.enabled",
    "policy.megatron_cfg.fp8_cfg.fp8",
    "policy.megatron_cfg.fp8_cfg.fp8_param",
    "policy.megatron_cfg.fp8_cfg.fp8_recipe",
    "policy.megatron_cfg.gradient_accumulation_fusion",
    "policy.megatron_cfg.linear_ce_fusion_chunk_size",
    "policy.megatron_cfg.moe_enable_deepep",
    "policy.megatron_cfg.moe_shared_expert_overlap",
    "policy.megatron_cfg.moe_token_dispatcher_type",
    "policy.megatron_cfg.recompute_granularity",
    "policy.megatron_cfg.recompute_modules",
    "policy.megatron_cfg.use_linear_ce_fusion_loss",
    "sft.val_at_end",
}

REPRODUCIBLE_SFT_DOCS = [
    Path("docs/releases/openmathreasoning/training.md"),
    Path("docs/releases/nemotron-math-v2/training.md"),
    Path("docs/releases/nemotronmathproofs/index.md"),
    Path("docs/releases/openmathinstruct2/training.md"),
]

REQUIRED_REPRODUCIBLE_DOC_OVERRIDES = PRESERVED_BY_WRAPPER_DEFAULTS - {
    # Release docs below all use the Megatron backend. The FSDP optimizer kwargs
    # are still preserved by the wrapper, but are not active reproduction
    # parameters for these documented commands.
    "policy.optimizer.kwargs.eps",
    "policy.optimizer.kwargs.lr",
    "policy.optimizer.kwargs.weight_decay",
}


def _task(data_format="input_output"):
    return NemoRLTask(
        model="/model",
        config_path="/opt/nemo-rl/examples/configs/sft.yaml",
        output_dir="/out",
        prompt_data="/train.jsonl",
        eval_data=None,
        num_gpus=8,
        num_nodes=1,
        expname="exp",
        disable_wandb=True,
        wandb_project="project",
        wandb_group=None,
        timeout="01:00:00",
        log_dir="/logs",
        env_variables={},
        backend="megatron",
        data_format=data_format,
        profile_step_range=None,
    )


def _cmd_overrides(cmd):
    return set(re.findall(r"\+\+([^=\s]+)=", cmd))


def _doc_sft_blocks(path):
    text = path.read_text()
    blocks = re.findall(r"```(?:bash|python)?\n(.*?)```", text, re.DOTALL)
    return [block for block in blocks if "nemo_rl sft" in block or "sft_nemo_rl(" in block]


def _flatten(value, prefix=""):
    if isinstance(value, dict):
        flattened = {}
        for key, nested_value in value.items():
            nested_prefix = f"{prefix}.{key}" if prefix else key
            flattened.update(_flatten(nested_value, nested_prefix))
        return flattened
    return {prefix: value}


def _load_flattened_config(path):
    return _flatten(yaml.safe_load(path.read_text()))


def _config_diffs(left, right):
    return {
        "common": {key for key in left.keys() & right.keys() if left[key] != right[key]},
        "legacy_only": set(left.keys() - right.keys()),
        "upstream_only": set(right.keys() - left.keys()),
    }


def test_nemo_rl_sft_all_legacy_upstream_default_diffs_are_classified():
    legacy = _load_flattened_config(LEGACY_SFT_CONFIG)
    upstream = _load_flattened_config(UPSTREAM_SFT_CONFIG)
    actual = _config_diffs(legacy, upstream)
    expected = {
        "common": (
            PRESERVED_COMMON_DEFAULTS | CLI_OR_RECIPE_CONTROLLED_COMMON | SCHEMA_OR_NON_TRAINING_BEHAVIOR_COMMON
        ),
        "legacy_only": PRESERVED_LEGACY_ONLY_DEFAULTS | LEGACY_ONLY_SCHEMA_OR_ALIASES,
        "upstream_only": UPSTREAM_ONLY_SCHEMA_OR_NEMO_RL_OWNED,
    }

    assert actual == expected


def test_nemo_rl_sft_preserves_nemo_skills_default_overrides():
    cmd = _task().get_cmd()
    overrides = _cmd_overrides(cmd)

    assert PRESERVED_BY_WRAPPER_DEFAULTS <= overrides

    expected_overrides = [
        "++sft.max_num_epochs=100000000",
        "++sft.max_num_steps=100000000",
        "++sft.val_period=0",
        "++sft.val_batches=1",
        "++sft.val_at_start=False",
        "++checkpointing.keep_top_k=50",
        "++checkpointing.save_period=100",
        "++policy.tokenizer.chat_template=null",
        "++policy.max_total_sequence_length=4096",
        "++policy.max_grad_norm=0.0",
        "++policy.sequence_packing.enabled=True",
        "++policy.megatron_cfg.layernorm_epsilon=1e-6",
        "++policy.megatron_cfg.moe_permute_fusion=false",
        "++policy.megatron_cfg.optimizer.lr=1e-6",
        "++policy.megatron_cfg.optimizer.min_lr=1e-6",
        "++policy.megatron_cfg.optimizer.weight_decay=0.01",
        "++policy.megatron_cfg.optimizer.adam_eps=1e-8",
        "++policy.megatron_cfg.scheduler.lr_decay_style=cosine",
        "++policy.megatron_cfg.scheduler.lr_decay_iters=\\${sft.max_num_steps}",
        "++policy.megatron_cfg.scheduler.lr_warmup_iters=0",
        "++policy.megatron_cfg.scheduler.lr_warmup_init=1.0e-6",
        "++policy.optimizer.kwargs.lr=1e-6",
        "++policy.optimizer.kwargs.weight_decay=0.01",
        "++policy.optimizer.kwargs.eps=1e-8",
        "++data.add_bos=false",
        "++data.add_eos=false",
        "++data.add_generation_prompt=false",
        "++data.num_workers=10",
    ]
    for override in expected_overrides:
        assert override in cmd

    assert cmd.index("++checkpointing.keep_top_k=50") < cmd.index("++data.validation=null")


def test_reproducible_sft_docs_list_compatibility_overrides():
    for relative_path in REPRODUCIBLE_SFT_DOCS:
        path = Path(__file__).parents[1] / relative_path
        for block in _doc_sft_blocks(path):
            overrides = _cmd_overrides(block)
            assert REQUIRED_REPRODUCIBLE_DOC_OVERRIDES <= overrides, relative_path


def test_nemo_rl_sft_messages_format_uses_default_chat_template():
    cmd = _task(data_format="messages").get_cmd()

    assert "++policy.tokenizer.chat_template=default" in cmd
    assert "++policy.tokenizer.chat_template=null" not in cmd


def test_detect_data_format(tmp_path):
    input_output = tmp_path / "input_output.jsonl"
    input_output.write_text(json.dumps({"input": "question", "output": "answer"}) + "\n")
    assert detect_data_format(str(input_output)) == "input_output"

    messages = tmp_path / "messages.jsonl"
    messages.write_text(json.dumps({"messages": [{"role": "user", "content": "question"}]}) + "\n")
    assert detect_data_format(str(messages)) == "messages"
