# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Schema adapters bridging external backends to NeMo-Skills' on-disk formats.

Lives outside ``nemo_skills.pipeline`` on purpose: the adapter runs *inside*
the backend's container (e.g. the Gym container) after each rollout, so it
has to be importable from ``nemo-skills-core`` alone. The pipeline package
gates on ``nemo_run`` and would fail to import there.
"""
