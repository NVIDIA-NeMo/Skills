# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Alpaca Audio: instruction-following with speech.

Speech-based instruction following derived from the Alpaca dataset.
Tests model's ability to:
- Follow spoken instructions
- Provide accurate responses based on audio context
- Handle diverse question types

Evaluated using LLM-as-a-judge.
"""

DATASET_GROUP = "speechlm"
METRICS_TYPE = "speechlm"
DEFAULT_SPLIT = "test"
GENERATION_ARGS = "++prompt_format=openai "

