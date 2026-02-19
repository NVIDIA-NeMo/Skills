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

from pathlib import Path

from nemo_skills.pipeline.utils.packager import (
    RepoMetadata,
    register_external_repo,
)

# Register repo so it gets packaged inside containers.
register_external_repo(
    RepoMetadata(name="my_benchmarks", path=Path(__file__).parents[2]),
    ignore_if_registered=True,
)

METRICS_TYPE = "math"
GENERATION_ARGS = "++prompt_config=generic/math ++eval_type=math"
