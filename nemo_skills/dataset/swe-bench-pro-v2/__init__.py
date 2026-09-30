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

# The default main container is glibc-based, so default to the non-Alpine split.
EVAL_SPLIT = "default.ubuntu"
METRICS_TYPE = "swe-bench"
GENERATION_MODULE = "nemo_skills.inference.eval.swebench_pro_v2"
GENERATION_ARGS = (
    "++multilingual=True ++dataset_type=swe_bench_pro_v2 "
    "++agent_framework=mini_swe_agent ++agent_max_turns=250 ++isolate_agent_network=True"
)
REQUIRES_DATA_DIR = True
