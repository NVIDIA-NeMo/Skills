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
"""Unit tests for the native AsyncOpenAI + aiohttp fast-path in BaseModel.

Covers the three feature-completion items:
  1. Native path is the DEFAULT for OpenAI-compatible providers and can be
     disabled with NEMO_SKILLS_OPENAI_AIOHTTP=0; non-OpenAI providers never
     use it.
  2. Reasoning content is read from BOTH `.reasoning` and `.reasoning_content`
     (litellm normalizes this for us; the native path sees vLLM's raw field).
  3. Streaming is routed through the native path and tolerates chunks with no
     choices.

These are pure unit tests -- no server or network access.
"""

import asyncio
import types

import openai
import pytest

from nemo_skills.inference.model.azure import AzureOpenAIModel
from nemo_skills.inference.model.base import BaseModel
from nemo_skills.inference.model.gemini import GeminiModel
from nemo_skills.inference.model.megatron import MegatronModel
from nemo_skills.inference.model.openai import OpenAIModel
from nemo_skills.inference.model.sglang import SGLangModel
from nemo_skills.inference.model.vllm import VLLMModel
from nemo_skills.inference.model.vllm_multimodal import VLLMMultimodalModel


# ---------------------------------------------------------------------------
# Item 2: reasoning extraction accepts both field names
# ---------------------------------------------------------------------------
def test_extract_reasoning_from_reasoning_content():
    obj = types.SimpleNamespace(reasoning_content="thinking")
    assert BaseModel._extract_reasoning(obj) == "thinking"


def test_extract_reasoning_from_reasoning():
    # New vLLM field name; only `.reasoning` present.
    obj = types.SimpleNamespace(reasoning="thinking")
    assert BaseModel._extract_reasoning(obj) == "thinking"


def test_extract_reasoning_prefers_reasoning_content():
    obj = types.SimpleNamespace(reasoning_content="primary", reasoning="secondary")
    assert BaseModel._extract_reasoning(obj) == "primary"


def test_extract_reasoning_absent_or_empty():
    assert BaseModel._extract_reasoning(types.SimpleNamespace()) is None
    assert BaseModel._extract_reasoning(types.SimpleNamespace(reasoning="", reasoning_content="")) is None


class _FakeUsage:
    completion_tokens = 5
    prompt_tokens = 3

    def model_dump(self):
        return {"completion_tokens": 5, "prompt_tokens": 3, "total_tokens": 8}


class _FakeChoice:
    def __init__(self, message):
        self.message = message
        self.finish_reason = "stop"

    def model_dump(self):
        return {"message": {"role": "assistant", "content": self.message.content}}


class _FakeResponse:
    def __init__(self, message):
        self.choices = [_FakeChoice(message)]
        self.usage = _FakeUsage()


def test_parse_chat_completion_reads_reasoning_field():
    # A response carrying ONLY `.reasoning` (as newer vLLM emits over the raw
    # OpenAI schema) must still populate reasoning_content in the result.
    inst = BaseModel.__new__(BaseModel)
    message = types.SimpleNamespace(content="the answer", reasoning="the thinking")
    result = inst._parse_chat_completion_response(_FakeResponse(message))
    assert result["generation"] == "the answer"
    assert result["reasoning_content"] == "the thinking"
    assert result["num_generated_tokens"] == 5


# ---------------------------------------------------------------------------
# Item 3: streaming chunk parsing (reasoning + empty-choices guard)
# ---------------------------------------------------------------------------
def test_process_chat_chunk_reads_reasoning_field():
    inst = BaseModel.__new__(BaseModel)
    delta = types.SimpleNamespace(content="hi", reasoning="step")
    chunk = types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta, finish_reason=None)])
    [result] = inst._process_chat_chunk(chunk)
    assert result["generation"] == "hi"
    assert result["reasoning_content"] == "step"


def test_stream_skips_empty_choices_chunk():
    inst = BaseModel.__new__(BaseModel)

    async def fake_stream():
        # usage-only / keepalive chunk with no choices -- must be skipped
        yield types.SimpleNamespace(choices=[])
        yield types.SimpleNamespace(
            choices=[types.SimpleNamespace(delta=types.SimpleNamespace(content="hello"), finish_reason=None)]
        )

    async def collect():
        return [r async for r in inst._stream_chat_chunks_async(fake_stream())]

    results = asyncio.run(collect())
    assert results == [{"generation": "hello"}]


# ---------------------------------------------------------------------------
# Item 1: which providers are eligible for the native fast-path
# ---------------------------------------------------------------------------
def test_native_eligibility_by_provider():
    assert BaseModel.SUPPORTS_NATIVE_OPENAI is False
    # OpenAI-compatible endpoints opt in.
    assert OpenAIModel.SUPPORTS_NATIVE_OPENAI is True
    assert VLLMModel.SUPPORTS_NATIVE_OPENAI is True
    assert SGLangModel.SUPPORTS_NATIVE_OPENAI is True
    # Non-OpenAI-shaped endpoints stay on litellm.
    assert AzureOpenAIModel.SUPPORTS_NATIVE_OPENAI is False
    assert GeminiModel.SUPPORTS_NATIVE_OPENAI is False
    assert MegatronModel.SUPPORTS_NATIVE_OPENAI is False
    assert VLLMMultimodalModel.SUPPORTS_NATIVE_OPENAI is False


class _DummyAioHttpClient:
    def __init__(self, *args, **kwargs):
        pass


class _DummyAsyncOpenAI:
    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def _mock_openai_client(monkeypatch):
    # Avoid needing the real openai[aiohttp] transport / a live endpoint.
    monkeypatch.setattr(openai, "AsyncOpenAI", _DummyAsyncOpenAI, raising=False)
    monkeypatch.setattr(openai, "DefaultAioHttpClient", _DummyAioHttpClient, raising=False)


def test_native_client_on_by_default(monkeypatch, _mock_openai_client):
    monkeypatch.delenv("NEMO_SKILLS_OPENAI_AIOHTTP", raising=False)
    model = VLLMModel(model="test-model", base_url="http://localhost:9999/v1")
    assert isinstance(model._async_openai_client, _DummyAsyncOpenAI)


def test_native_client_opt_out(monkeypatch, _mock_openai_client):
    monkeypatch.setenv("NEMO_SKILLS_OPENAI_AIOHTTP", "0")
    model = VLLMModel(model="test-model", base_url="http://localhost:9999/v1")
    assert model._async_openai_client is None


def test_native_client_off_for_ineligible_provider(monkeypatch, _mock_openai_client):
    monkeypatch.delenv("NEMO_SKILLS_OPENAI_AIOHTTP", raising=False)

    class _NoNative(VLLMModel):
        SUPPORTS_NATIVE_OPENAI = False

    model = _NoNative(model="test-model", base_url="http://localhost:9999/v1")
    assert model._async_openai_client is None


class _CapturingAsyncOpenAI:
    """Captures the kwargs passed to chat.completions.create."""

    def __init__(self, *args, **kwargs):
        self.captured_kwargs = None
        outer = self

        class _Completions:
            async def create(self, **kw):
                outer.captured_kwargs = kw
                return _FakeResponse(types.SimpleNamespace(content="ok"))

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


def test_native_params_drop_none_and_litellm_only(monkeypatch):
    # The native path must not forward litellm-only keys (allowed_openai_params)
    # nor explicit None values (which the SDK would serialize as JSON null,
    # unlike litellm which omits them).
    monkeypatch.delenv("NEMO_SKILLS_OPENAI_AIOHTTP", raising=False)
    monkeypatch.setattr(openai, "AsyncOpenAI", _CapturingAsyncOpenAI, raising=False)
    monkeypatch.setattr(openai, "DefaultAioHttpClient", _DummyAioHttpClient, raising=False)

    model = VLLMModel(model="test-model", base_url="http://localhost:9999/v1")
    assert isinstance(model._async_openai_client, _CapturingAsyncOpenAI)

    asyncio.run(
        model.generate_async(
            prompt=[{"role": "user", "content": "hi"}],
            tokens_to_generate=8,
            temperature=0.0,
            top_p=0.95,
            random_seed=None,  # -> seed=None in the builder, must be dropped
            stop_phrases=None,  # -> stop=None, must be dropped
            tools=None,  # -> tools=None, must be dropped
            response_format=None,  # -> response_format=None, must be dropped
        )
    )

    captured = model._async_openai_client.captured_kwargs
    assert captured is not None
    assert "allowed_openai_params" not in captured
    assert all(v is not None for v in captured.values()), f"None leaked: {captured}"
    # explicit-None builder fields were dropped, not sent as null
    for dropped in ("seed", "stop", "tools", "response_format", "top_logprobs"):
        assert dropped not in captured
    assert captured["model"] == "test-model"


def test_pydantic_response_format_falls_back_to_litellm(monkeypatch):
    # A pydantic BaseModel response_format (structured output) must NOT go through
    # the native client -- chat.completions.create() rejects a BaseModel class
    # ("must use .parse() instead"). litellm accepts it, so we fall back.
    import litellm
    from pydantic import BaseModel

    monkeypatch.delenv("NEMO_SKILLS_OPENAI_AIOHTTP", raising=False)
    monkeypatch.setattr(openai, "AsyncOpenAI", _CapturingAsyncOpenAI, raising=False)
    monkeypatch.setattr(openai, "DefaultAioHttpClient", _DummyAioHttpClient, raising=False)

    litellm_called = {"hit": False}

    async def fake_acompletion(**kwargs):
        litellm_called["hit"] = True
        return _FakeResponse(types.SimpleNamespace(content='{"ok": true}'))

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    class _RF(BaseModel):
        answer: str

    model = VLLMModel(model="test-model", base_url="http://localhost:9999/v1")
    assert isinstance(model._async_openai_client, _CapturingAsyncOpenAI)  # native is available...

    asyncio.run(
        model.generate_async(
            prompt=[{"role": "user", "content": "hi"}],
            tokens_to_generate=8,
            temperature=0.0,
            top_p=0.95,
            response_format=_RF,
        )
    )

    # ...but it was NOT used for the structured-output request; litellm handled it.
    assert model._async_openai_client.captured_kwargs is None
    assert litellm_called["hit"] is True
