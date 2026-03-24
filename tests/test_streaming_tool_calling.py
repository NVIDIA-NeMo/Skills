# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for streaming tool-calling with tokenizer-based token counting."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_skills.inference.model.base import EndpointType
from nemo_skills.inference.model.tool_call import ToolCallingWrapper


class FakeTokenizer:
    """Tokenizer that splits on whitespace for predictable token counts."""

    def encode(self, text):
        if not text:
            return []
        return text.split()


def _make_wrapper(tokenizer=None):
    """Create a ToolCallingWrapper with mocked model and tool_manager."""
    model = MagicMock()
    model.tokenizer = tokenizer or FakeTokenizer()

    with patch.object(ToolCallingWrapper, "__init__", lambda self, *a, **kw: None):
        wrapper = ToolCallingWrapper.__new__(ToolCallingWrapper)

    wrapper.model = model
    wrapper.tool_manager = AsyncMock()
    wrapper.tool_manager.list_all_tools = AsyncMock(return_value=[])
    wrapper.schema_overrides = {}
    wrapper.schema_mappings = {}
    wrapper.max_tool_calls = -1
    return wrapper


def _collect(agen):
    """Collect all items from an async generator."""

    async def _inner():
        return [item async for item in agen]

    return asyncio.run(_inner())


_PATCH_TOOLS = patch(
    "nemo_skills.inference.model.tool_call.format_tool_list_by_endpoint_type",
    return_value=([], {}),
)


def test_stream_no_tool_calls():
    """Streaming without tool calls yields chunks + final with tokenizer-counted tokens."""
    wrapper = _make_wrapper()

    async def mock_stream(*args, **kwargs):
        yield {"generation": "Hello ", "finish_reason": None}
        yield {"generation": "world", "finish_reason": "stop"}

    wrapper.model.generate_async = AsyncMock(return_value=mock_stream())

    with _PATCH_TOOLS:
        results = _collect(
            wrapper._stream_single(
                prompt=[{"role": "user", "content": "hi"}],
                endpoint_type=EndpointType.chat,
            )
        )

    final = results[-1]
    assert final["type"] == "final"
    assert final["generation"] == "Hello world"
    assert final["num_generated_tokens"] == 2  # "Hello world" → 2 whitespace tokens
    assert final["num_tool_calls"] == 0


def test_stream_with_tool_call():
    """Streaming with tool calls yields tool_calls/tool_results events and counts tokens correctly."""
    wrapper = _make_wrapper()
    call_count = 0

    async def mock_stream_factory(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:

            async def gen():
                yield {
                    "generation": "",
                    "tool_calls": {
                        "index": 0,
                        "id": "call_abc",
                        "type": "function",
                        "function": {"name": "calc", "arguments": '{"expr": "2+2"}'},
                    },
                    "finish_reason": "tool_calls",
                }

            return gen()
        else:

            async def gen():
                yield {"generation": "The answer is 4", "finish_reason": "stop"}

            return gen()

    wrapper.model.generate_async = AsyncMock(side_effect=mock_stream_factory)

    async def mock_execute_tool_calls(tool_calls, request_id, endpoint_type):
        return [{"role": "tool", "tool_call_id": "call_abc", "content": "4"}]

    wrapper._execute_tool_calls = mock_execute_tool_calls

    with _PATCH_TOOLS:
        results = _collect(
            wrapper._stream_single(
                prompt=[{"role": "user", "content": "what is 2+2?"}],
                endpoint_type=EndpointType.chat,
            )
        )

    types = [r.get("type") for r in results]
    assert "tool_calls" in types
    assert "tool_results" in types

    final = [r for r in results if r.get("type") == "final"][0]
    assert final["num_tool_calls"] == 1
    assert final["num_generated_tokens"] == 4  # "The answer is 4"
    assert final["generation"] == "The answer is 4"


def test_stream_tool_response_tokens_counted():
    """Tool response content tokens are subtracted from the tokens_to_generate budget."""
    wrapper = _make_wrapper()
    call_count = 0

    async def mock_stream_factory(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # tokens_to_generate starts at 10; model generation = "hi" (1 token),
        # tool response = "result value" (2 tokens) → budget after first turn = 10 - 1 - 2 = 7
        if call_count == 1:

            async def gen():
                yield {
                    "generation": "",
                    "tool_calls": {
                        "index": 0,
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    },
                    "finish_reason": "tool_calls",
                }

            return gen()
        else:

            async def gen():
                yield {"generation": "done", "finish_reason": "stop"}

            return gen()

    wrapper.model.generate_async = AsyncMock(side_effect=mock_stream_factory)

    async def mock_execute_tool_calls(tool_calls, request_id, endpoint_type):
        return [{"role": "tool", "tool_call_id": "c1", "content": "result value"}]

    wrapper._execute_tool_calls = mock_execute_tool_calls

    with _PATCH_TOOLS:
        results = _collect(
            wrapper._stream_single(
                prompt=[{"role": "user", "content": "go"}],
                endpoint_type=EndpointType.chat,
                tokens_to_generate=10,
            )
        )

    final = [r for r in results if r.get("type") == "final"][0]
    # model tokens: "" (0) + "done" (1) = 1
    # tool response "result value" = 2 tokens (whitespace split)
    # total accounted: 3, budget remaining: 10 - 0 - 2 - 1 = 7 (not tested directly)
    assert final["num_generated_tokens"] == 1  # only model-generated text
    assert final["num_tool_calls"] == 1


def test_stream_max_tool_calls_stops_loop():
    """max_tool_calls cap prevents executing more tool calls than allowed."""
    wrapper = _make_wrapper()
    wrapper.max_tool_calls = 1
    call_count = 0

    async def mock_stream_factory(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        async def gen():
            yield {
                "generation": "",
                "tool_calls": {
                    "index": 0,
                    "id": f"c{call_count}",
                    "type": "function",
                    "function": {"name": "f", "arguments": "{}"},
                },
                "finish_reason": "tool_calls",
            }

        return gen()

    wrapper.model.generate_async = AsyncMock(side_effect=mock_stream_factory)
    execute_count = 0

    async def mock_execute_tool_calls(tool_calls, request_id, endpoint_type):
        nonlocal execute_count
        execute_count += 1
        return [{"role": "tool", "tool_call_id": tool_calls[0]["id"], "content": "ok"}]

    wrapper._execute_tool_calls = mock_execute_tool_calls

    with _PATCH_TOOLS:
        results = _collect(
            wrapper._stream_single(
                prompt=[{"role": "user", "content": "go"}],
                endpoint_type=EndpointType.chat,
            )
        )

    final = [r for r in results if r.get("type") == "final"][0]
    assert final["num_tool_calls"] == 1
    assert final["finish_reason"] == "tool_call_limit_reached"
    assert execute_count == 1  # second tool call was never executed


def test_stream_no_tokenizer_raises():
    """Streaming without a tokenizer raises RuntimeError."""
    wrapper = _make_wrapper()
    wrapper.model.tokenizer = None

    with pytest.raises(RuntimeError, match="Tokenizer is required"):
        _collect(
            wrapper._stream_single(
                prompt=[{"role": "user", "content": "hi"}],
                endpoint_type=EndpointType.chat,
            )
        )
