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

"""AudioBench judge tasks.

Tasks in this category require LLM-as-a-judge evaluation:
- Speech QA: Open-ended questions about audio content
- Emotion/Sentiment: Classification tasks requiring semantic understanding
- Audio captioning: Descriptive text generation
- Music understanding: Questions about musical content

These tasks use an LLM judge to evaluate the quality and correctness of responses.
"""

DATASET_GROUP = "speechlm"
METRICS_TYPE = "speechlm"
DEFAULT_SPLIT = "test"
GENERATION_ARGS = "++prompt_format=openai "

