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

from recipes.sherloc.inference.sherloc_utils.patch_processor import PatchProcessor


def test_pure_additions_in_adjacent_lines_of_different_files_are_retained():
    patch = "\n".join(
        [
            "--- a/first.py",
            "+++ b/first.py",
            "@@ -2,0 +3 @@",
            "+first addition",
            "--- a/second.py",
            "+++ b/second.py",
            "@@ -3,0 +4 @@",
            "+second addition",
        ]
    )

    locations = PatchProcessor.extract_locations_from_patch(patch)

    assert [(location["file_path"], location["start_line"]) for location in locations] == [
        ("first.py", 2),
        ("second.py", 3),
    ]


def test_no_newline_marker_does_not_advance_patch_coordinates():
    patch = "\n".join(
        [
            "--- a/example.py",
            "+++ b/example.py",
            "@@ -1 +1 @@",
            "-old",
            r"\ No newline at end of file",
            "+new",
            r"\ No newline at end of file",
        ]
    )

    locations = PatchProcessor.extract_locations_from_patch(patch)

    assert locations == [
        {
            "file_path": "example.py",
            "start_line": 1,
            "end_line": 1,
            "raw": "example.py:L1-L1",
        }
    ]


def test_insertion_before_first_line_uses_one_based_location():
    patch = "\n".join(
        [
            "--- a/example.py",
            "+++ b/example.py",
            "@@ -0,0 +1 @@",
            "+new first line",
        ]
    )

    locations = PatchProcessor.extract_locations_from_patch(patch)

    assert locations == [
        {
            "file_path": "example.py",
            "start_line": 1,
            "end_line": 1,
            "raw": "example.py:L1-L1",
        }
    ]
