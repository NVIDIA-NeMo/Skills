# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

# Settings that define how evaluation should be done by default.
#
# task_type=ASR (not ASR_LEADERBOARD): scoring uses the AppTek-specific
# normalization ``apptek_callcenter`` rather than the HF Open ASR Leaderboard
# normalizer, so the leaderboard task code (with its multi-reference logic)
# does not apply here.

REQUIRES_DATA_DIR = True
DEFAULT_SPLIT = "test"

METRICS_TYPE = "audio"
EVAL_ARGS = "++eval_type=audio ++eval_config.normalization_mode=apptek_callcenter"
GENERATION_ARGS = "++prompt_format=openai ++enable_audio=true"
