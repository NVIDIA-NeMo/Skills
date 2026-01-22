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

"""AudioBench non-judge tasks dataset configuration.

This dataset includes ASR, translation, and other tasks that use
automatic metrics (WER, BLEU, WER-PC) instead of judge evaluation.

NO JUDGE REQUIRED - Metrics computed automatically from model outputs.
"""

# Dataset configuration - CRITICAL: needed for audio to work
DATASET_GROUP = "speechlm"
METRICS_TYPE = "audio"

# Evaluation settings
EVAL_ARGS = "++eval_type=audio ++eval_config.normalization_mode=audiobench"

# Generation settings - OpenAI format for audio-language models
GENERATION_ARGS = "++prompt_format=openai ++enable_audio=true"
