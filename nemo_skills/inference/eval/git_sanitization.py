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

# A subshell keeps pipefail local to cleanup. Explicit && checks are required:
# Bash ignores errexit inside a subshell used in the launcher's && chain.
# Use Git's ref APIs so loose and packed refs behave identically. Preserve the
# working tree and index, including legitimate environment setup changes.
_SAFE_GIT_CLEANUP = """(
set -o pipefail &&
NEMO_BASE_COMMIT=$(git rev-parse --verify 'HEAD^{commit}') &&
git update-ref refs/heads/main "$NEMO_BASE_COMMIT" &&
git symbolic-ref HEAD refs/heads/main &&
git for-each-ref --format='%(refname)' |
while IFS= read -r ref; do
    if [ "$ref" != refs/heads/main ]; then
        git update-ref --no-deref -d "$ref" || exit 1
    fi
done &&
git reflog expire --expire=now --all &&
git gc --prune=now --aggressive &&
test "$(git rev-parse --verify 'HEAD^{commit}')" = "$NEMO_BASE_COMMIT" &&
git cat-file -e "$NEMO_BASE_COMMIT^{tree}"
)"""


def rewrite_legacy_git_cleanup(pre_commands: str) -> str:
    """Replace the known dataset cleanup without rewriting arbitrary shell code.

    Invoke this in the shared launcher so every agent framework gets the fix
    without regenerating prepared datasets. Unrelated setup commands are kept
    verbatim; this is not a general-purpose shell parser or container sanitizer.
    """
    return pre_commands.replace(_LEGACY_GIT_CLEANUP, _SAFE_GIT_CLEANUP)
