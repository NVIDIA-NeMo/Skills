# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

# Benchmark group: run all multilingual AIME25 variants (same pattern as bfcl_v3 / mmau-pro).
DATASET_GROUP = "math"

VARIANTS = [
    "aime25-de-prompt-de",
    "aime25-de-prompt-en",
    "aime25-es-prompt-en",
    "aime25-es-prompt-es",
    "aime25-fr-prompt-en",
    "aime25-fr-prompt-fr",
    "aime25-ja-prompt-en",
    "aime25-ja-prompt-ja",
]

IS_BENCHMARK_GROUP = True

SCORE_MODULE = "nemo_skills.dataset.aime25-X.aime25_x_group_score"

BENCHMARKS = {f"aime25-X.{name}": {} for name in VARIANTS}
