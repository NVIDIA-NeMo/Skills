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

"""Reference edit locations extracted from gold patches.

SHERLOC predictions are scored against the locations that the reference fix of
an issue touches. That fix ships as a unified diff, so this module maps a diff
to the files and line ranges it modifies. Ranges are reported in the coordinate
system of the pre-patch file, which is the version of the repository the agent
inspects, so predicted and reference locations are directly comparable.
"""

import re
from typing import Any, Dict, List


class PatchProcessor:
    """Git patch processing utilities."""

    @staticmethod
    def extract_locations_from_patch(patch: str, exclude_new_files: bool = True) -> List[Dict[str, Any]]:
        """Extract changed line ranges from a git patch, in pre-patch coordinates.

        Line numbers refer to the file as it exists before the patch is applied,
        which is the version the agent inspects, so references and predictions
        share one coordinate system. Adjacent ranges in the same file are merged.

        Args:
            patch: Unified diff text to parse
            exclude_new_files: When True, files the patch creates contribute no
                locations, since such a file has no pre-patch content to point at

        Returns:
            List of dicts with the keys ``file_path``, ``start_line``,
            ``end_line`` and ``raw`` (the ``path:L<start>-L<end>`` rendering).
        """
        if not patch:
            return []

        locations = []
        current_file = None
        is_new_file = False
        original_line = 0
        new_line = 0

        for line in patch.splitlines():
            # File path from --- line
            if line.startswith("--- "):
                file_path = line[4:]
                if file_path == "/dev/null":
                    # This is a new file, wait for +++ line
                    is_new_file = True
                    current_file = None
                else:
                    if file_path.startswith(("a/", "b/")):
                        file_path = file_path[2:]
                    current_file = file_path
                    is_new_file = False

            # File path from +++ line (for new files)
            elif line.startswith("+++ "):
                if is_new_file:
                    file_path = line[4:]
                    if file_path.startswith(("a/", "b/")):
                        file_path = file_path[2:]
                    current_file = file_path

            # Hunk header
            elif line.startswith("@@ "):
                # Parse: @@ -original_start[,original_count] +new_start[,new_count] @@
                m = re.match(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
                if m:
                    original_line = int(m.group(1))
                    new_line = int(m.group(3))

            elif current_file and line:
                # Defensive: the --- and +++ branches above already resolve /dev/null
                if current_file == "/dev/null":
                    continue
                if line == r"\ No newline at end of file":
                    continue

                # Track changes in original file
                if line.startswith("-") and not line.startswith("---"):
                    # Line removed from original - this is a change location
                    locations.append(
                        {
                            "file_path": current_file,
                            "start_line": original_line,
                            "end_line": original_line,
                            "raw": f"{current_file}:L{original_line}-L{original_line}",
                        }
                    )
                    original_line += 1
                elif line.startswith("+") and not line.startswith("+++"):
                    # Line added
                    if is_new_file:
                        # For new files, track line 1 as the change location
                        # We only add this once per new file
                        if not exclude_new_files and not any(loc["file_path"] == current_file for loc in locations):
                            locations.append(
                                {
                                    "file_path": current_file,
                                    "start_line": 1,
                                    "end_line": 1,
                                    "raw": f"{current_file}:L1-L1",
                                }
                            )
                    else:
                        # For existing files, track where the addition would be inserted
                        if (
                            not locations
                            or locations[-1]["file_path"] != current_file
                            or locations[-1]["end_line"] != original_line - 1
                        ):
                            # Pure addition at current position in original
                            locations.append(
                                {
                                    "file_path": current_file,
                                    "start_line": original_line,
                                    "end_line": original_line,
                                    "raw": f"{current_file}:L{original_line}-L{original_line}",
                                }
                            )
                    new_line += 1
                else:
                    # Context line - advances both counters
                    if not is_new_file:  # Only advance for existing files
                        original_line += 1
                    new_line += 1

        # Merge adjacent locations
        merged = []
        for loc in locations:
            if (
                merged
                and loc["file_path"] == merged[-1]["file_path"]
                and loc["start_line"] <= merged[-1]["end_line"] + 1
            ):
                # Extend the previous location
                merged[-1]["end_line"] = max(merged[-1]["end_line"], loc["end_line"])
                merged[-1]["raw"] = f"{merged[-1]['file_path']}:L{merged[-1]['start_line']}-L{merged[-1]['end_line']}"
            else:
                merged.append(loc)

        return merged
