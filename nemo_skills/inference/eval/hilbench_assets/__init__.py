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

"""Verbatim assets for the HIL-Bench SWE evaluator (``nemo_skills.inference.eval.hilbench``).

These files are *not* imported as Python modules; they are read as text and written out
unchanged at runtime:

* ``ask_human_tool/`` - the SWE-agent ``ask_human`` tool bundle (config + client + installer)
  copied into ``/root/SWE-agent/tools/ask_human``.
* ``hil_eval_in_container.py`` - the self-contained evaluator executed inside each task image.
* ``ask_human_server.py`` - the Flask judge server launched on the host.

Keeping them as standalone files (rather than escaped string literals embedded in the module)
makes them editable, lintable, and syntax-highlightable as real source.
"""
