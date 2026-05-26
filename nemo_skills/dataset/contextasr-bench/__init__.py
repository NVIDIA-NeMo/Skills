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

"""ContextASR-Bench: Contextual ASR evaluation benchmark.

Evaluates ASR models across three context settings:
- Contextless: Plain transcription
- Coarse-grained: Domain label provided as context
- Fine-grained: Domain label + entity list provided as context

Metrics: WER, NE-WER (entity-focused WER with fuzzy matching), NE-FNR (entity miss rate)

Dataset: https://huggingface.co/datasets/MrSupW/ContextASR-Bench
Paper: ContextASR-Bench (English Speech subset, 15,326 samples, ~188 hours)
"""

REQUIRES_DATA_DIR = True
IS_BENCHMARK_GROUP = True
SCORE_MODULE = "nemo_skills.dataset.contextasr-bench.contextasr_score"

BENCHMARKS = {
    "contextasr-bench.contextless": {},
    "contextasr-bench.coarse": {},
    "contextasr-bench.fine": {},
}
