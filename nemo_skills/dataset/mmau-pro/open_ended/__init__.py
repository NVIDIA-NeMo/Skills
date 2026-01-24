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

# Open-ended questions evaluated with LLM judge (Qwen)
METRICS_TYPE = "mmau_pro_open_ended"
SCORE_MODULE = "nemo_skills.evaluation.metrics.mmau_pro_metrics"
GENERATION_ARGS = "++prompt_format=openai ++enable_audio=true"

# Judge configuration for open-ended evaluation using NVIDIA API
JUDGE_PIPELINE_ARGS = {
    "model": "qwen/qwen2.5-7b-instruct",
    "server_type": "openai",
    "server_address": "https://integrate.api.nvidia.com/v1",
}
JUDGE_ARGS = "++prompt_config=judge/mmau-pro ++generation_key=judgement"
