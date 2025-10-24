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

"""
todo: We are working on providing the data files that are necessary to run IOI25 evaluation.
"""

# settings that define how evaluation should be done by default (all can be changed from cmdline)
GENERATION_ARGS = "++prompt_config=generic/default ++eval_type=ioi"
DATASET_GROUP = "code"
METRICS_TYPE = "ioi"

# environment variables required by this benchmark
SANDBOX_ENV_VARS = [
    "UWSGI_PROCESSES=1024",
    "UWSGI_CPU_AFFINITY=8",
    "UWSGI_CHEAPER=1023",
    "NUM_WORKERS=1",
    "STATEFUL_SANDBOX=0",
]
