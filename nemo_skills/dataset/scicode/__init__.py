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
METRICS_TYPE = "scicode"
# Use the multistep ("scientist-annotated background is provided, write code directly")
# prompt to match the prompt Artificial Analysis actually sends to the endpoint.
GENERATION_ARGS = "++prompt_config=eval/scicode/background ++eval_type=scicode"
GENERATION_MODULE = "nemo_skills.inference.eval.scicode"
REQUIRES_SANDBOX = True
EVAL_SPLIT = "test"  # 65 problems / 288 subproblems, matching AA's SciCode methodology
