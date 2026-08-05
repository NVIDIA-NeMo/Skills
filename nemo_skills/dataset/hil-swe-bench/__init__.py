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

# HiL-Bench SWE subset only (SQL is intentionally not included).
#
# settings that define how evaluation should be done by default (all can be changed from cmdline)
EVAL_SPLIT = "default"
METRICS_TYPE = "hil-swe-bench"
# Generation uses hil-bench SWE-agent; grading uses Harbor tests/test.sh via hil_bench.py.
GENERATION_MODULE = "nemo_skills.inference.eval.hil_bench"
GENERATION_ARGS = "++agent_framework=swe_agent ++dataset_type=hil_bench ++agent_max_turns=100"
# Harbor task trees are large; keep them under --data_dir (do not package with every job).
REQUIRES_DATA_DIR = True
