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

DATASET_GROUP = "math"
METRICS_TYPE = "math"
GENERATION_ARGS = "++prompt_config=generic/math ++eval_type=math"
# Judge configuration: Use the AnswerAutoGrader prompt.
# Recommended model: Gemini 2.5 Pro (or similar strong reasoner)
JUDGE_ARGS = "++prompt_config=judge/imo_answerbench ++generation_key=judgement ++inference.reasoning_effort=dynamic"

JUDGE_PIPELINE_ARGS = {
    "generation_type": "math_judge",
    "model": "gemini-2.5-pro",
    "server_type": "gemini",
}
