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

# Benchmarks
# https://artificialanalysis.ai/methodology/intelligence-benchmarking
IS_BENCHMARK_GROUP = True

SCORE_MODULE = "nemo_skills.dataset.aai.aai_score"

BENCHMARKS = {
    # Reasoning and Knowledge benchmarks
    "mmlu-pro": {
        "GENERATION_ARGS": "++prompt_config=eval/aai/mcq-10choices ++inference.temperature=0.0",
        # can add "NUM_CHUNKS": N to parallelize
    },
    "hle": {
        "GENERATION_ARGS": "++inference.temperature=0.0",
        "JUDGE_ARGS": "++prompt_config=judge/hle ++generation_key=judgement",
    },
    # Science benchmarks
    "gpqa": {
        "GENERATION_ARGS": "++prompt_config=eval/aai/mcq-4choices ++inference.temperature=0.0",
    },
    # Math benchmarks
    "aime25": {
        "GENERATION_ARGS": "++prompt_config=eval/aai/math ++inference.temperature=0.0",
        "NUM_SAMPLES": 10,
    },
    # Coding benchmarks
    "scicode": {
        "GENERATION_ARGS": "++inference.temperature=0.0",
        "EVAL_SPLIT": "test_aai",
        "NUM_SAMPLES": 3,
    },
    "livecodebench": {
        "GENERATION_ARGS": "++prompt_config=eval/aai/livecodebench ++inference.temperature=0.0",
        "EVAL_SPLIT": "test_v5_2407_2412",
        "NUM_SAMPLES": 3,
    },
    # Instruction following benchmarks
    "ifbench": {
        "GENERATION_ARGS": "++prompt_config=generic/default ++inference.temperature=0.0",
        "EVAL_SPLIT": "test",
        "NUM_SAMPLES": 5,
    },
    # Long context reasoning benchmarks. To fully match the AA-LCR setting, you need to use judge with Qwen3-235B-A22B-Instruct-2507 in aalcr/__init__.py
    "aalcr": {
        "GENERATION_ARGS": "++inference.temperature=0.0",
        "NUM_SAMPLES": 4,
    },
}
