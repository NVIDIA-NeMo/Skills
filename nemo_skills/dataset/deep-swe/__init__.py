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
EVAL_SPLIT = "default"
METRICS_TYPE = "deep-swe"
# Agent generation reuses SWE-bench (swe_agent / mini_swe_agent / openhands / gold_patch).
# Grading uses Harbor tests/test.sh via deepswe.py (not the SWE-bench harness).
GENERATION_MODULE = "nemo_skills.inference.eval.deepswe"
GENERATION_ARGS = (
    "++multilingual=True ++dataset_type=deep_swe "
    "++agent_max_turns=400 ++use_agent_timeouts=True ++use_verifier_timeouts=True "
    "++execution_backend=opensandbox"
)
# Harbor task trees are large; keep them under --data_dir (do not package with every job).
REQUIRES_DATA_DIR = True
