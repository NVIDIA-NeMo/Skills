# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
METRICS_TYPE = "frontierscience-olympiad"
GENERATION_ARGS = "++prompt_config=generic/default ++eval_type=math"
EVAL_SPLIT = "all"
JUDGE_PIPELINE_ARGS = {
    "model": "o3-mini-2025-01-31",
    "server_type": "openai",
    "server_address": "https://api.openai.com/v1",
}
JUDGE_ARGS = "++prompt_config=judge/frontierscience-olympiad ++generation_key=judgement ++add_generation_stats=False"
