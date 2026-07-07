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
# settings that define how evaluation should be done by default (all can be changed from cmdline)

METRICS_TYPE = "multichoice"

# Few-shot is disabled by default. To enable 5-shot evaluation, add the examples_type override.
# The escaping differs depending on where you add it:
#
#   In GENERATION_ARGS (this file):  ++examples_type=\\'{examples_type}\\'
#   On the command line (with surrounding quotes): "++examples_type=\\\\\\\"{examples_type}\\\\\\\""
#
# The prompt code formats {examples_type} per row as the user message in the prompt.
# At inference time, the `examples_type` field is written into each dataset entry by prepare.py
# (e.g. "mmlu_prox_few_shot_de"). This triggers the dynamic few-shot loader in
# nemo_skills/prompt/few_shot_examples/mmlu_prox.py, which downloads the first 5 examples from
# the li-lab/MMLU-ProX validation split for the appropriate language.
GENERATION_ARGS = "++prompt_config=multilingual/mmlu-prox ++eval_type=multichoice"
