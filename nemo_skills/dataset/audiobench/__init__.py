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

"""AudioBench: A comprehensive benchmark for speech and audio language models.

AudioBench evaluates models across multiple tasks:
- ASR (Automatic Speech Recognition)
- Translation (speech-to-text translation)
- Speech QA (question answering based on audio)
- Audio understanding (emotion, gender, accent recognition, etc.)

The benchmark is organized into two main categories:
- nonjudge: Tasks evaluated with automatic metrics (WER, BLEU)
- judge: Tasks requiring LLM-as-a-judge evaluation
"""

DATASET_GROUP = "speechlm"
IS_BENCHMARK_GROUP = True
SCORE_MODULE = "nemo_skills.evaluation.metrics.audio_metrics"

# Top-level benchmarks: evaluate all judge or all nonjudge datasets
BENCHMARKS = {
    "audiobench.nonjudge": {},
    "audiobench.judge": {},
}
