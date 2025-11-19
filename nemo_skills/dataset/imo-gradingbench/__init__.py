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

DATASET_GROUP = "judge"
METRICS_TYPE = "math"
# This dataset is for evaluating the judge itself, or using a judge to evaluate.
# If it's a meta-benchmark, we might want to see if the model's judgment matches the expected reward.
GENERATION_ARGS = "++prompt_config=judge/math ++eval_type=math"
