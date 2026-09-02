# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""NTT-SMOKE: compact English NemotronTranscribe smoke evaluation.

``ntt-smoke.en`` is a single mixed manifest with subtask metadata. This keeps
generation/evaluation reporting consolidated while still reporting metrics by
subtask through ``subset_for_metrics`` and NTT-specific aggregate metrics.
"""

REQUIRES_DATA_DIR = True
IS_BENCHMARK_GROUP = True
SCORE_MODULE = "nemo_skills.dataset.ntt-smoke.ntt_smoke_metrics"

BENCHMARKS = {
    "ntt-smoke.en": {},
}
