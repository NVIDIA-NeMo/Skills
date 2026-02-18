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
EVAL_SPLIT = "test"
DATASET_GROUP = "code"
METRICS_TYPE = "math"

# Use DSBench evaluator (extends MathEvaluator) with relaxed extraction and case-insensitive MCQ:
# 1. \boxed{X} - standard LaTeX (caught by search_boxed)
# 2. "The final answer is X" or "\boxed=X" - (caught by regex)
# 3. Fallback to relaxed_equal for case-insensitive MCQ and dict/list comparison
GENERATION_ARGS = (
    '++prompt_config=generic/dsbench-da '
    '++eval_type=dsbench' # similar to math evaluator but uses case insensitive matching for MCQ, and parsing for dict and list 
    '++eval_config.relaxed=true '
)

# # Recommend running LLM judge to verify dicts and lists correctly 
# JUDGE_PIPELINE_ARGS = {
#     "generation_type": "math_judge",
#     "model": "gpt-4.1",
#     "server_type": "openai",
#     "server_address": "https://api.openai.com/v1",
# }
