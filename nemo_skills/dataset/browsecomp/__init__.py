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
# BrowseComp answers cannot be compared symbolically, so correctness is decided by an LLM judge.
# The judge output is graded by the browsecomp evaluator (++eval_type=browsecomp in JUDGE_ARGS),
# which writes a `judge_correct` boolean that BrowseCompMetrics aggregates.
METRICS_TYPE = "browsecomp"
GENERATION_ARGS = "++prompt_config=eval/browsecomp"
EVAL_SPLIT = "test"

# BrowseComp requires a judge model to evaluate factual accuracy.
# Setting an OpenAI judge by default, but this can be overridden from the command line for a locally hosted model.
JUDGE_PIPELINE_ARGS = {
    "model": "o3-mini-2025-01-31",
    "server_type": "openai",
    "server_address": "https://api.openai.com/v1",
}
JUDGE_ARGS = (
    "++prompt_config=judge/browsecomp ++generation_key=judgement ++eval_type=browsecomp ++add_generation_stats=False"
)
