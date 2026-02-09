# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

# SysBench dataset configuration for NeMo Skills.

# This dataset evaluates how well models follow the system instructions in the
# SysBench benchmark (https://github.com/PKU-Baichuan-MLSystemLab/SysBench).

# settings that define how evaluation should be done by default (all can be changed from cmdline)
DATASET_GROUP = "chat"
METRICS_TYPE = "sysbench"
DEFAULT_SPLIT = "test"
EVAL_SPLIT = "test"

# Use custom generation module for multi-turn evaluation
GENERATION_MODULE = "nemo_skills.inference.eval.sysbench"
GENERATION_ARGS = "++prompt_format=openai ++generation_key=generation"

JUDGE_PIPELINE_ARGS = {
    "model": "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "server_type": "vllm",
    "server_gpus": 8,  # Number of GPUs for judge server
}

JUDGE_ARGS = "++prompt_config=judge/sys-bench ++generation_key=judgement ++add_generation_stats=False ++inference.tokens_to_generate=65536"

