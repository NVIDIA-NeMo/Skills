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

import json
from types import SimpleNamespace

import pytest

from recipes.sherloc.inference.sherloc_utils.context_manager import ContextManager


def test_oversized_first_turn_preview_has_complete_stats():
    turns = [{"_input_tokens": 100}, {"_tool_tokens": 10}]

    preview = ContextManager.get_truncation_preview(turns, max_seq_length=100, tokens_to_generate=10)

    assert "Keep: Turn 0" in preview
    assert "Remove: Turns [1]" in preview


def test_context_check_requires_turns():
    cfg = SimpleNamespace(
        max_seq_length=100,
        context_safety_margin=0.9,
        inference=SimpleNamespace(tokens_to_generate=10),
    )

    with pytest.raises(KeyError):
        ContextManager.check_context_before_generation({}, cfg)


def test_truncation_target_respects_safety_margin():
    cfg = SimpleNamespace(
        max_seq_length=1000,
        context_safety_margin=0.9,
        inference=SimpleNamespace(tokens_to_generate=100),
    )
    turns = [{"_input_tokens": 400}, {"_tool_tokens": 300}, {"_tool_tokens": 300}]
    safe_max = int(cfg.max_seq_length * cfg.context_safety_margin)

    kept, _ = ContextManager.first_and_recent_truncate(turns, safe_max, cfg.inference.tokens_to_generate)

    assert ContextManager.count_dialogue_tokens(kept) <= safe_max - cfg.inference.tokens_to_generate
    assert ContextManager.check_context_before_generation({"turns": kept}, cfg)[0] is True


@pytest.mark.parametrize(
    "repeated_call",
    [
        {"view_file": {"path": "pkg/cache.py"}},
        {"tool": "view_file", "path": "pkg/cache.py"},
    ],
)
def test_loop_intervention_supports_both_tool_call_shapes(repeated_call):
    loop_info = {"repeated_call": json.dumps(repeated_call), "total_repetitions": 3}

    turns = ContextManager.inject_loop_intervention([{"inputs": "continue"}], loop_info)
    intervention = turns[0]["inputs"]

    assert "pkg/cache.py" in intervention
    assert "codebase_search" in intervention
    assert "repo_tree" in intervention
    assert "connected_tree" in intervention
    assert "grep or find" not in intervention
    assert "list_directory" not in intervention
