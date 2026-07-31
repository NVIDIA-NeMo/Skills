# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

# settings that define how evaluation should be done by default (all can be changed from cmdline)
# Debian/Ubuntu containers work with the standard NeMo-Skills image. Run the
# default.alpine split separately with Dockerfile.nemo-skills.alpine.
EVAL_SPLIT = "default.ubuntu"
METRICS_TYPE = "swe-atlas-qna"
GENERATION_MODULE = "nemo_skills.inference.eval.swe_atlas_qna"
GENERATION_ARGS = (
    "++agent_framework=mini_swe_agent "
    "++agent_config=eval/swe-atlas-qna/mini-swe-agent/default "
    "++agent_max_turns=250 "
    "++evaluate=False"
)

# SWE-Atlas-QnA uses an LLM to grade each answer against its task-specific rubric.
# Claude Opus 4.5 is the judge used by the official benchmark. The server address
# must point to an OpenAI-compatible endpoint that serves this model.
JUDGE_PIPELINE_ARGS = {
    "generation_module": "nemo_skills.inference.swe_atlas_qna_judge",
    "model": "azure/anthropic/claude-opus-4-5",
    "server_type": "openai",
    "server_address": "https://inference-api.nvidia.com",
}
JUDGE_ARGS = (
    "++prompt_config=judge/swe-atlas-qna "
    "++generation_key=judgement "
    "++inference.temperature=1.0 "
    "++inference.tokens_to_generate=2048 "
    "++add_generation_stats=False"
)
