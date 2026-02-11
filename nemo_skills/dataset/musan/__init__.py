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

"""MUSAN: A Music, Speech, and Noise Corpus

MUSAN is a corpus of music, speech, and noise recordings designed for training
models for voice activity detection and music/speech discrimination.

DOWNLOAD OPTIONS:

1. HuggingFace (default - INCOMPLETE):
   - 774 samples, 5h 4m total
   - Noise: 728 samples (78% complete)
   - Fast, no API key needed

2. Kaggle (RECOMMENDED - COMPLETE): ✓
   - 10.3 GB, 2,016 WAV files
   - Noise: 930 files (99.8% complete!)
   - Music: 660 files, Speech: 426 files
   - Requires Kaggle API key (one-time setup)

3. OpenSLR (official - COMPLETE):
   - 11 GB, full dataset
   - No API key needed

Reference:
    David Snyder, Guoguo Chen, and Daniel Povey
    "MUSAN: A Music, Speech, and Noise Corpus"
    arXiv:1510.08484, 2015
"""

REQUIRES_DATA_DIR = True
IS_BENCHMARK_GROUP = True
SCORE_MODULE = "nemo_skills.evaluation.metrics.audio_metrics"
METRICS_TYPE = "audio"

# Evaluation settings
EVAL_ARGS = "++eval_type=audio "

# Generation settings - OpenAI format for audio-language models
GENERATION_ARGS = "++prompt_format=openai "

# Benchmark - single test.jsonl contains all noise samples at top level
BENCHMARKS = {
    "musan": {},
}
