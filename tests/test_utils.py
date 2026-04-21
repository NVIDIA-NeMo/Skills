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

"""Unit tests for nemo_skills.utils."""

from nemo_skills.utils import parse_reasoning


def test_parse_reasoning_splits_on_end_tag():
    sample = {"generation": "<think>reasoning here</think>\n\nFinal answer"}
    parse_reasoning(sample)
    assert sample["generation"] == "Final answer"
    assert sample["_generation_finished_thinking"] is True
    assert sample["_full_generation"] == "<think>reasoning here</think>\n\nFinal answer"


def test_parse_reasoning_keeps_text_after_last_end_tag():
    sample = {"generation": "<think>one</think>middle<think>two</think>answer"}
    parse_reasoning(sample)
    assert sample["generation"] == "answer"
    assert sample["_generation_finished_thinking"] is True


def test_parse_reasoning_preserves_generation_when_end_tag_missing():
    """No </think> → leave generation untouched.

    Covers two real cases: (1) server-side reasoning parser already stripped
    the tags and returned clean text, and (2) non-reasoning model accidentally
    had parse_reasoning=True. Wiping the generation loses data in both cases.
    """
    original = "The answer is 42."
    sample = {"generation": original}
    parse_reasoning(sample)
    assert sample["generation"] == original
    assert sample["_generation_finished_thinking"] is False
    assert sample["_full_generation"] == original


def test_parse_reasoning_preserves_partial_think_when_truncated():
    """Truncated reasoning (no closing tag) → keep partial text, flag as unfinished."""
    original = "<think>reasoning that got cut of"
    sample = {"generation": original}
    parse_reasoning(sample)
    assert sample["generation"] == original
    assert sample["_generation_finished_thinking"] is False


def test_parse_reasoning_custom_end_string():
    sample = {"generation": "<reason>x</reason>\nanswer"}
    parse_reasoning(sample, end_reasoning_string="</reason>")
    assert sample["generation"] == "answer"
    assert sample["_generation_finished_thinking"] is True


def test_parse_reasoning_custom_generation_key():
    sample = {"out": "<think>x</think>\nanswer"}
    parse_reasoning(sample, generation_key="out")
    assert sample["out"] == "answer"
    assert sample["_out_finished_thinking"] is True
    assert sample["_full_out"] == "<think>x</think>\nanswer"


def test_parse_reasoning_non_string_no_op():
    sample = {"generation": ["not", "a", "string"]}
    parse_reasoning(sample)
    assert sample["generation"] == ["not", "a", "string"]
    assert "_generation_finished_thinking" not in sample
