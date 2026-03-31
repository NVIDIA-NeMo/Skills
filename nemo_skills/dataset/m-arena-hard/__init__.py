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
METRICS_TYPE = "arena"
# using judgement directly in metrics, no need for special evaluation
GENERATION_ARGS = "++prompt_config=generic/default"

JUDGE_ARGS = "++sanitize_generations=true"

JUDGE_PIPELINE_ARGS = {
    "generation_module": "nemo_skills.inference.eval.arena_judge",
    "model": "openai/openai/gpt-5.2",
    "server_type": "openai",
    "server_address": "https://inference-api.nvidia.com/v1",
}
