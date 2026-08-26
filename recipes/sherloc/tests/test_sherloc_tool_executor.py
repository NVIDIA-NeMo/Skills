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

from types import SimpleNamespace

import pytest

from recipes.sherloc.inference.sherloc_utils.tool_executor import ToolExecutor


def _executor():
    executor = ToolExecutor.__new__(ToolExecutor)
    executor.cfg = SimpleNamespace(max_view_lines=1000, show_line_counts=False)
    executor._count_tokens = lambda text: len(text)
    return executor


def test_codebase_search_paths_are_relative_to_repository_root():
    repo = {
        "instance_id": "example",
        "structure": {
            "pkg": {
                "cache.py": {
                    "classes": [],
                    "functions": [],
                    "text": ["def make_cache_key():", "    return 'cache-key'"],
                }
            }
        },
    }

    output, _ = _executor()._execute_codebase_search_tool({"query": "cache-key"}, repo)

    assert "File: pkg/cache.py" in output
    assert "structure/pkg/cache.py" not in output


def test_view_tool_truncates_and_rejects_non_integer_ranges():
    executor = _executor()
    executor.cfg.max_view_lines = 10
    repo = {
        "structure": {
            "big.py": {
                "classes": [],
                "functions": [],
                "text": [f"line {line_number}" for line_number in range(50)],
            }
        }
    }

    output, _ = executor._execute_view_tool({"path": "big.py", "view_range": [1, 50]}, repo)
    invalid, _ = executor._execute_view_tool({"path": "big.py", "view_range": ["1", "50"]}, repo)
    invalid_mapping, _ = executor._execute_view_tool({"path": "big.py", "view_range": {"start": 1}}, repo)

    assert "TRUNCATED" in output
    assert invalid.startswith("Error: Invalid view_range format")
    assert invalid_mapping.startswith("Error: Invalid view_range format")


def test_unexpected_snapshot_errors_propagate():
    executor = _executor()

    with pytest.raises(KeyError):
        executor._execute_repo_tree_tool({})
    with pytest.raises(KeyError):
        executor._execute_connected_tree_tool({}, {})
    with pytest.raises(KeyError):
        executor._execute_codebase_search_tool({"query": "cache"}, {})
