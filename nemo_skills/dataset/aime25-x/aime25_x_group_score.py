# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


def compute_score(combined_metrics: dict) -> dict:
    """Merge metrics from all `aime25-X.*` sub-benchmarks into one JSON blob.

    Per-variant files are merged before this runs; we return the combined dict as the
    group-level `metrics.json`. Add weighted averages here if you need a single headline number.
    """
    return dict(combined_metrics)
