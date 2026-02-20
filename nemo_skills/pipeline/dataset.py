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

"""Cluster-aware dataset loading.

Re-exports get_dataset_module from core. The cluster-aware functionality
(mount-path resolution, SSH downloads) is handled by core's get_dataset_module
when a cluster_config is provided — pipeline-only deps are lazily imported
inside those code paths.
"""

from nemo_skills.dataset.utils import get_dataset_module  # noqa: F401
