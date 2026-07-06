# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

# MGSM (Multilingual Grade School Math): GSM8K test problems translated into 10
# languages plus English. Answers are integers, graded by exact match after
# extracting \boxed{...}. Per-language accuracy is reported via subset_for_metrics.
#
# The default prompt uses a single English boxed instruction (en-CoT setting).
# To score native-language reasoning as well, switch METRICS_TYPE to
# "math_multilingual" (uses target_language, already populated below) and use a
# target-language prompt.
METRICS_TYPE = "math"
GENERATION_ARGS = "++prompt_config=generic/general-boxed ++eval_type=math"
