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

# settings that define how evaluation should be done by default (all can be changed from cmdline)
PROMPT_CONFIG = "generic/default"
DATASET_GROUP = "long-context"
METRICS_TYPE = "aalcr"
# using judgement directly in metrics, no need for special evaluation
GENERATION_ARGS = "++prompt_config=generic/default"

JUDGE_PIPELINE_ARGS = {
    "model": "gpt-4.1",
    "server_type": "openai",
    "server_address": "https://api.openai.com/v1",
}

JUDGE_ARGS = "++prompt_config=judge/aalcr ++generation_key=judgement ++add_generation_stats=False"

# AA-LCR official judge model.
# JUDGE_PIPELINE_ARGS = {
#     "model": "/hf_models/Qwen3-235B-A22B-Instruct-2507",
#     "server_type": "sglang",
#     "server_gpus": 8,
# }
