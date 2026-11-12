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

"""Compatibility for Git cleanup embedded in prepared SWE-bench datasets."""

from nemo_skills.dataset.utils import get_swe_sanitization_command

# Prepared datasets can contain this sequence even when it is absent from the
# source checkout. Deleting packed-refs directly also deletes main when its ref
# is packed, leaving HEAD unresolved before the agent can start.
_LEGACY_GIT_CLEANUP = (
    "rm -f .git/packed-refs && "
    "find .git/refs/heads -type f ! -name 'main' -delete && "
    "rm -rf .git/refs/tags .git/refs/remotes && "
    "git reflog expire --expire=now --all && "
    "git gc --prune=now --aggressive"
)


def rewrite_legacy_git_cleanup(pre_commands: str) -> str:
    """Replace the legacy dataset cleanup procedure with the new, more robust version."""
    return pre_commands.replace(_LEGACY_GIT_CLEANUP, get_swe_sanitization_command())
