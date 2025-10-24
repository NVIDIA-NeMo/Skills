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

# settings that define how evaluation should be done by default (all can be changed from cmdline)
DATASET_GROUP = "math"
METRICS_TYPE = "simpleqa"
GENERATION_ARGS = "++prompt_config=generic/default ++eval_type=math"
EVAL_SPLIT = "verified"

# SimpleQA requires judge model for evaluating factual accuracy
# Setting openai judge by default, but can be overridden from command line for a locally hosted model
# Using o3-mini-20250131 as recommended for factual evaluation tasks

JUDGE_PIPELINE_ARGS = {
    "model": "o3-mini-20250131",
    "server_type": "openai",
    "server_address": "https://api.openai.com/v1",
}


JUDGE_ARGS = "++prompt_config=judge/simpleqa ++generation_key=judgement ++add_generation_stats=False"
