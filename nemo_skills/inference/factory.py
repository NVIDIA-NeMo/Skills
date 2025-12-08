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

from enum import Enum


class GenerationType(str, Enum):
    generate = "generate"
    math_judge = "math_judge"
    check_contamination = "check_contamination"


GENERATION_MODULE_MAP = {
    GenerationType.generate: "nemo_skills.inference.generate",
    GenerationType.math_judge: "nemo_skills.inference.llm_math_judge",
    GenerationType.check_contamination: "nemo_skills.inference.check_contamination",
}
