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

"""Unit tests for OpenAIModel._build_chat_request_params."""

from unittest.mock import MagicMock, patch

import pytest

from nemo_skills.inference.model.openai import OpenAIModel


@pytest.fixture
def model():
    with patch.object(OpenAIModel, "__init__", lambda self, **kwargs: None):
        m = OpenAIModel.__new__(OpenAIModel)
        m.model = "some-self-hosted-model"
        return m


COMMON_KWARGS = dict(
    tokens_to_generate=512,
    temperature=0.7,
    top_p=0.95,
    top_k=-1,
    min_p=0.0,
    repetition_penalty=1.0,
    random_seed=42,
    stop_phrases=[],
    timeout=300,
    top_logprobs=None,
    stream=False,
    reasoning_effort=None,
    tools=None,
    response_format=None,
)


def test_extra_body_passthrough(model):
    """extra_body should be forwarded to litellm for self-hosted endpoints."""
    extra_body = {"chat_template_kwargs": {"thinking": True}}
    params = model._build_chat_request_params(
        messages=[{"role": "user", "content": "hello"}],
        extra_body=extra_body,
        **COMMON_KWARGS,
    )
    assert params["extra_body"] == extra_body


def test_extra_body_none_omitted(model):
    """extra_body=None should not appear in the returned params."""
    params = model._build_chat_request_params(
        messages=[{"role": "user", "content": "hello"}],
        extra_body=None,
        **COMMON_KWARGS,
    )
    assert "extra_body" not in params


def test_extra_body_empty_dict_omitted(model):
    """extra_body={} should not appear in the returned params."""
    params = model._build_chat_request_params(
        messages=[{"role": "user", "content": "hello"}],
        extra_body={},
        **COMMON_KWARGS,
    )
    assert "extra_body" not in params


def test_extra_body_multiple_fields(model):
    """Multiple extra_body fields are forwarded intact."""
    extra_body = {"chat_template_kwargs": {"thinking": True}, "guided_json": {"type": "object"}}
    params = model._build_chat_request_params(
        messages=[{"role": "user", "content": "hello"}],
        extra_body=extra_body,
        **COMMON_KWARGS,
    )
    assert params["extra_body"] == extra_body


def test_standard_params_unaffected(model):
    """Adding extra_body does not disturb standard request params."""
    params = model._build_chat_request_params(
        messages=[{"role": "user", "content": "hello"}],
        extra_body={"chat_template_kwargs": {"thinking": True}},
        **COMMON_KWARGS,
    )
    assert params["temperature"] == COMMON_KWARGS["temperature"]
    assert params["top_p"] == COMMON_KWARGS["top_p"]
    assert params["max_completion_tokens"] == COMMON_KWARGS["tokens_to_generate"]
