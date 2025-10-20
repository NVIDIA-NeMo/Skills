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

from nemo_skills.version import __version__

# only used in ns setup command to initialize with defaults
_containers = {
    "trtllm": "nvcr.io/nvidia/tensorrt-llm/release:1.0.0",
    "vllm": "vllm/vllm-openai:v0.10.1.1",
    "sglang": "lmsysorg/sglang:v0.5.3rc1-cu126",
    "megatron": "dockerfile:dockerfiles/Dockerfile.megatron",
    "sandbox": "dockerfile:dockerfiles/Dockerfile.sandbox",
    "nemo-skills": "dockerfile:dockerfiles/Dockerfile.nemo-skills",
    "verl": "dockerfile:dockerfiles/Dockerfile.verl",
    "nemo-rl": "dockerfile:dockerfiles/Dockerfile.nemo-rl",
}
