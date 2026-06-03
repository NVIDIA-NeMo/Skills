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

"""CS-FLEURS: massively multilingual code-switched ASR evaluation.

Code-switched speech recognition across the four CS-FLEURS test sets, each a
sub-benchmark scored with per-language CER/WER (CER for scriptio-continua matrix
languages, WER otherwise):

- ``cs-fleurs.read``        14 X-English pairs, human-read speech (the paper's
                            intended, human-validated benchmarking set)
- ``cs-fleurs.mms``         45 X-English pairs, concatenative MMS-TTS speech
- ``cs-fleurs.xtts-test1``  16 X-English pairs, generative XTTS-v2 speech
- ``cs-fleurs.xtts-test2``  60 language pairs, generative XTTS-v2 speech

Dataset: https://huggingface.co/datasets/byan/cs-fleurs (CC-BY-NC-4.0)
Paper: CS-FLEURS: A Massively Multilingual and Code-Switched Speech Dataset
       (https://arxiv.org/abs/2509.14161)
"""

REQUIRES_DATA_DIR = True
IS_BENCHMARK_GROUP = True
SCORE_MODULE = "nemo_skills.dataset.cs-fleurs.audio_score"

BENCHMARKS = {
    "cs-fleurs.read": {},
    "cs-fleurs.mms": {},
    "cs-fleurs.xtts-test1": {},
    "cs-fleurs.xtts-test2": {},
}
