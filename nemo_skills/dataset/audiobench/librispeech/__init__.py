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

"""LibriSpeech ASR evaluation.

Clean speech ASR evaluation on LibriSpeech test sets:
- test-clean: Clean speech recordings
- test-other: More challenging speech with various acoustic conditions

Evaluated using Word Error Rate (WER).
"""

DATASET_GROUP = "speechlm"
METRICS_TYPE = "speechlm"
DEFAULT_SPLIT = "test"
EVAL_ARGS = "++eval_type=audiobench "
GENERATION_ARGS = "++prompt_format=openai "

