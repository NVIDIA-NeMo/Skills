# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# Google FACTS Grounding benchmark
# Evaluates LLMs' ability to generate long-form responses grounded in a provided context document.
# See: https://huggingface.co/datasets/google/FACTS-grounding-public

METRICS_TYPE = "facts_grounding"
GENERATION_ARGS = "++prompt_config=generic/facts_grounding ++inference.tokens_to_generate=8192"

# LLM judge evaluation using NVIDIA Inference API.
# ``model`` seeds server/client setup in the nemo-skills pipeline; the judge
# task itself spins up one client per entry in ``judge_models`` (see
# ``nemo_skills.inference.eval.facts_grounding_judge``). All three judges are
# reachable via the same base URL, so only the ``model`` field varies per call.
JUDGE_PIPELINE_ARGS = {
    "generation_module": "nemo_skills.inference.eval.facts_grounding_judge",
    "model": "gcp/google/gemini-3.1-pro-preview",
    "server_type": "openai",
    "server_address": "https://inference-api.nvidia.com/v1",
}
# Default 3-judge ensemble — closest available counterparts to the
# google-facts reference's gemini-3-pro / gpt-5.2 / claude-opus-4-5.
# Override via ``++judge_models=[...]`` on the CLI.
JUDGE_ARGS = (
    "++prompt_config=judge/facts_grounding "
    "++generation_key=judgement "
    "++add_generation_stats=False "
    "++inference.temperature=0.5 "
    "++inference.tokens_to_generate=8192 "
    "++judge_models=[gcp/google/gemini-3.1-pro-preview,azure/openai/gpt-5.2,aws/anthropic/claude-opus-4-5]"
)
