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

# IHEval benchmark group: 9 sub-benchmarks across rule-following, task-execution, safety, tool-use.

SPLITS = [
    "rule_following_single",
    "rule_following_multi",
    "task_execution_verb_extract",
    "task_execution_translation",
    "task_execution_lang_detect",
    "safety_hijack",
    "safety_extract",
    "tool_use_webpage",
    "tool_use_slack_user",
]

IS_BENCHMARK_GROUP = True

BENCHMARKS = {f"iheval.{split}": {} for split in SPLITS}

SCORE_MODULE = "nemo_skills.dataset.iheval.iheval_score"
