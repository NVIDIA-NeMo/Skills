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

"""Earnings call transcription (Earnings-21).

Long-form ASR evaluation on corporate earnings call recordings.
Tests model performance on:
- Financial/business terminology
- Multiple speakers
- Long audio segments

Evaluated using Word Error Rate (WER).
"""

DATASET_GROUP = "speechlm"
METRICS_TYPE = "speechlm"
DEFAULT_SPLIT = "test"
EVAL_ARGS = "++eval_type=audiobench "
GENERATION_ARGS = "++prompt_format=openai "

