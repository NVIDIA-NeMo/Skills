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
EVAL_SPLIT = "test"
METRICS_TYPE = "browsecomp"
GENERATION_ARGS = "++prompt_format=openai"

JUDGE_PIPELINE_ARGS = {
    # "model": "gpt-4o-mini",
    "model": "openai/openai/gpt-4o-mini",
    "server_type": "openai",
    "server_address": "https://inference-api.nvidia.com/v1",
}

# JUDGE_PIPELINE_ARGS = {
#     "model": "/hf_models/Qwen3-235B-A22B-Instruct-2507",
#     "server_type": "sglang",
#     "server_gpus": 8,
# }

JUDGE_ARGS = "++prompt_config=judge/browsecomp ++generation_key=judgement ++add_generation_stats=False"
