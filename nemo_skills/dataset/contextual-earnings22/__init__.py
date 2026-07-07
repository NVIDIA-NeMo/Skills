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

"""Contextual Earnings-22: contextual ASR benchmark on earnings-call clips.

Evaluates ASR models across three context settings:
- Contextless: Plain transcription (no context provided)
- Local:       Per-clip keyword list (only keywords actually spoken in the clip)
- Global:      Per-call keyword inventory (includes distractors not in the clip)

Metrics: WER + corpus-level keyword Precision / Recall / F1, with TP/FP/FN
counted by edit-distance-aligned exact match (paper Section 3).

Dataset: https://huggingface.co/datasets/argmaxinc/contextual-earnings22
Paper:   https://arxiv.org/abs/2604.07354 (Contextual Earnings-22, Argmax 2025)
"""

REQUIRES_DATA_DIR = True
IS_BENCHMARK_GROUP = True
SCORE_MODULE = "nemo_skills.dataset.contextual-earnings22.earnings22_score"

BENCHMARKS = {
    "contextual-earnings22.contextless": {},
    "contextual-earnings22.local": {},
    "contextual-earnings22.global": {},
}
