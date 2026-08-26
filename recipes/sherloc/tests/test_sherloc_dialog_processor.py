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

from recipes.sherloc.inference.sherloc_utils.dialog_processor import DialogProcessor


def _locations_block(body):
    """Wrap raw location lines in a <locations> block."""
    return f"<locations>\n{body}\n</locations>"


def test_extract_locations_range_form():
    result = DialogProcessor._extract_locations(_locations_block("a.py:L10-L20"))
    assert result["locations"] == [{"file_path": "a.py", "start_line": 10, "end_line": 20, "raw": "a.py:L10-L20"}]


def test_extract_locations_single_line_form():
    result = DialogProcessor._extract_locations(_locations_block("a.py:L42"))
    location = result["locations"][0]
    assert location["file_path"] == "a.py"
    assert location["start_line"] == 42
    assert location["end_line"] == 42


def test_extract_locations_zero_padded_numbers():
    result = DialogProcessor._extract_locations(_locations_block("a.py:L05-L07"))
    assert result["locations"][0]["start_line"] == 5
    assert result["locations"][0]["end_line"] == 7


def test_extract_locations_preserves_unparsable_line():
    result = DialogProcessor._extract_locations(_locations_block("not a location"))
    assert result["locations"] == [{"raw": "not a location"}]


def test_extract_locations_skips_comments():
    result = DialogProcessor._extract_locations(_locations_block("# a note\na.py:L1-L2"))
    assert len(result["locations"]) == 1
    assert result["locations"][0]["file_path"] == "a.py"


def test_extract_locations_multiple_entries():
    result = DialogProcessor._extract_locations(_locations_block("a.py:L1-L2\nb.py:L3-L4"))
    assert [entry["file_path"] for entry in result["locations"]] == ["a.py", "b.py"]


@pytest.mark.parametrize("body", ["", "\n", "# only a comment"])
def test_extract_locations_empty_block(body):
    assert DialogProcessor._extract_locations(_locations_block(body))["locations"] == []


def test_extract_locations_missing_block():
    assert DialogProcessor._extract_locations("no tags here")["locations"] == []


def test_extract_findings_returns_block_text():
    text = "<findings>\n- Root cause: locale omitted from the cache key.\n</findings>"
    assert DialogProcessor.extract_findings(text) == "- Root cause: locale omitted from the cache key."


def test_extract_findings_missing_block():
    assert DialogProcessor.extract_findings("no tags here") is None


def test_extract_findings_empty_block():
    assert DialogProcessor.extract_findings("<findings>\n\n</findings>") is None


def test_extract_response_includes_findings_with_locations():
    text = (
        "<findings>\n- Root cause: locale omitted from the cache key.\n</findings>\n"
        "<locations>\na.py:L10-L20\n</locations>"
    )
    response = DialogProcessor.extract_response(text)
    assert response["type"] == "locations"
    assert response["findings"] == "- Root cause: locale omitted from the cache key."
    assert response["locations"][0]["file_path"] == "a.py"


def test_extract_response_locations_without_findings():
    response = DialogProcessor.extract_response(_locations_block("a.py:L1-L2"))
    assert response["findings"] is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ('<tool_call>{"tool": "view_file", "path": "a.py"}</tool_call>', {"tool": "view_file", "path": "a.py"}),
        ('<tool_call>{"tool": "view_file", "path": "a.py"}', {"tool": "view_file", "path": "a.py"}),
        ('<tool_call>{"tool": "repo_tree"}}</tool_call>', {"tool": "repo_tree"}),
        ('<tool_call><![CDATA[{"tool": "repo_tree"}]]></tool_call>', {"tool": "repo_tree"}),
        ('<tool_call><view_file>{"path": "a.py"}</view_file></tool_call>', {"tool": "view_file", "path": "a.py"}),
        ("<tool_call><repo_tree></repo_tree></tool_call>", {"tool": "repo_tree"}),
    ],
)
def test_extract_response_recovers_tool_calls(text, expected):
    response = DialogProcessor.extract_response(text)
    assert response is not None
    for key, value in expected.items():
        assert response["tool_call"][key] == value


def test_extract_response_nested_tool_format():
    text = '<tool_call>{"view_file": {"path": "a.py", "view_range": [1, 50]}}</tool_call>'
    tool_call = DialogProcessor.extract_response(text)["tool_call"]
    assert tool_call["tool"] == "view_file"
    assert tool_call["path"] == "a.py"
    assert tool_call["view_range"] == [1, 50]


def test_extract_response_prefers_tool_call_over_locations():
    text = '<tool_call>{"tool": "repo_tree"}</tool_call>\n<locations>\na.py:L1-L2\n</locations>'
    assert DialogProcessor.extract_response(text)["tool_call"]["tool"] == "repo_tree"


def test_extract_response_returns_locations_when_no_tool_call():
    response = DialogProcessor.extract_response(_locations_block("a.py:L1-L2"))
    assert response["type"] == "locations"
    assert response["locations"][0]["file_path"] == "a.py"


@pytest.mark.parametrize("text", ["", "   ", "just some prose about the parser"])
def test_extract_response_returns_none_when_unrecoverable(text):
    config = SimpleNamespace(enable_implicit_tool_detection=False)
    assert DialogProcessor.extract_response(text, config) is None


def test_extract_response_implicit_detection_is_config_gated():
    text = "reasoning</think> please open src/a.py now"
    off = SimpleNamespace(enable_implicit_tool_detection=False)
    on = SimpleNamespace(enable_implicit_tool_detection=True, file_extensions=["py", "cfg"], common_words_filter=[])
    assert DialogProcessor.extract_response(text, off) is None
    assert DialogProcessor.extract_response(text, on)["tool_call"]["tool"] == "view_file"
