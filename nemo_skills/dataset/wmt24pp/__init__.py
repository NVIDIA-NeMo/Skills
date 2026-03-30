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

# WMT24++ translation benchmark group.
#
# Sub-benchmarks:
#   wmt24pp.sent  — Sentence-level translation (one sentence at a time, BLEU/COMET).
#   wmt24pp.doc   — Document-level translation (one document at a time, SEGALE scoring).
#   wmt24pp.long  — Long-context translation (all documents per language, SEGALE scoring).
#
# Default: ns eval wmt24pp runs only wmt24pp.sent
# (preserves existing behavior before adding doc and long variants).
# To run document-level: ns eval wmt24pp.doc
# To run all: ns eval wmt24pp.sent,wmt24pp.doc,wmt24pp.long

IS_BENCHMARK_GROUP = True

# Default behavior: only sentence-level (preserves existing behavior).
BENCHMARKS = {
    "wmt24pp.sent": {},
}
