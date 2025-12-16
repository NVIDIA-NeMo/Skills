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

"""AudioBench judge tasks dataset configuration.

This dataset includes tasks that require LLM-based evaluation such as:
- Audio captioning
- Spoken question answering
- Audio understanding and reasoning

These tasks require an LLM judge for evaluation, matching MMAU-Pro evaluation setup.
"""

# Dataset configuration - CRITICAL: needed for audio to work
DATASET_GROUP = "speechlm"
METRICS_TYPE = "audio"
DEFAULT_SPLIT = "test"
GENERATION_ARGS = "++prompt_format=openai "
EVAL_ARGS = "++eval_type=audio "

# Judge configuration matching AudioBench official implementation
# Using Llama-3.1-70B with vllm (can be overridden in run scripts)
JUDGE_PIPELINE_ARGS = {
    "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "server_type": "vllm",
    "server_gpus": 8,
    "server_args": "--max-model-len 8192 --gpu-memory-utilization 0.95",
}
JUDGE_ARGS = "++prompt_config=judge/audiobench ++generation_key=judgement"
