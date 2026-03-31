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

# Benchmark group: run all multilingual GPQA variants.
DATASET_GROUP = "multichoice"

VARIANTS = [
    "gpqa-de-prompt-de",
    "gpqa-de-prompt-en",
    "gpqa-es-prompt-en",
    "gpqa-es-prompt-es",
    "gpqa-fr-prompt-en",
    "gpqa-fr-prompt-fr",
    "gpqa-ja-prompt-en",
    "gpqa-ja-prompt-ja",
]

IS_BENCHMARK_GROUP = True

SCORE_MODULE = "nemo_skills.dataset.gpqa-X.gpqa_x_group_score"

BENCHMARKS = {f"gpqa-X.{name}": {} for name in VARIANTS}
