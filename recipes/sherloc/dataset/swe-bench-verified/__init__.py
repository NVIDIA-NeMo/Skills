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

"""SWE-Bench Verified wired up for SHERLOC diagnostic localization.

The task is not to produce a patch but to name the edit sites: generation runs the SHERLOC
agent loop over the repository's read-only inspection tools, and evaluation compares the
predicted locations against the gold patch, reporting file-level precision/recall/F1 and
line-range containment metrics.
"""

# settings that define how evaluation should be done by default (all can be changed from cmdline)
DATASET_GROUP = "code"
METRICS_TYPE = "recipes.sherloc.metrics.sherloc::SherlocMetrics"
GENERATION_MODULE = "recipes.sherloc.inference.sherloc"
GENERATION_ARGS = "++prompt_config=recipes/sherloc/prompt/eval/sherloc/system.yaml"
EVAL_ARGS = "++eval_type=recipes.sherloc.evaluation.sherloc::eval_sherloc"
REQUIRES_SANDBOX = False
