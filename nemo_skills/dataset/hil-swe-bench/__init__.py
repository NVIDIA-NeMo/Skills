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
# Metrics / generation module will be finalized with inference/eval/hil_bench.py.
# swe-bench metrics keys are compatible with Harbor reward "resolved" for now.
METRICS_TYPE = "swe-bench"
GENERATION_MODULE = "nemo_skills.inference.eval.hil_bench"
GENERATION_ARGS = ""
# Harbor task trees are large; keep them under --data_dir (do not package with every job).
REQUIRES_DATA_DIR = True
