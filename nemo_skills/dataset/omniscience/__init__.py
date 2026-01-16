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
DATASET_GROUP = "math"
METRICS_TYPE = "omniscience"
GENERATION_ARGS = "++prompt_config=eval/aai/omni ++parse_reasoning=True "
EVAL_SPLIT = "text"

JUDGE_PIPELINE_ARGS = {
    "model": "gemini-2.5-flash-preview-09-2025",
    "server_type": "gemini",
    "server_address": "https://generativelanguage.googleapis.com",
}
JUDGE_ARGS = "++prompt_config=judge/aa-omni-judge ++generation_key=judgement ++add_generation_stats=False"
