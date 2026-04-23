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

"""LibriSpeechMix benchmark group for overlapped ASR and speaker-attributed ASR."""

REQUIRES_DATA_DIR = True
IS_BENCHMARK_GROUP = True
SCORE_MODULE = "nemo_skills.dataset.librispeechmix.librispeechmix_score"

_MODES = ("asr", "sa-asr")
_SPLITS = ("dev-clean", "test-clean")
_MIXES = ("1mix", "2mix", "3mix")

BENCHMARKS = {
    f"librispeechmix.{mode}-{split_name}-{mix_name}": {}
    for mode in _MODES
    for split_name in _SPLITS
    for mix_name in _MIXES
}
