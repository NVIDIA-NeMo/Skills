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

import pytest

from nemo_skills.inference.eval.swebench import (
    build_claude_code_settings,
    get_claude_code_api_base,
    transform_claude_code_request,
)


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        ("http://127.0.0.1:8000/v1", "http://127.0.0.1:8000"),
        ("http://127.0.0.1:8000/v1/", "http://127.0.0.1:8000"),
        ("https://gateway.example.com", "https://gateway.example.com"),
        ("https://gateway.example.com/api", "https://gateway.example.com/api"),
    ],
)
def test_get_claude_code_api_base(api_base, expected):
    assert get_claude_code_api_base(api_base) == expected


def test_build_claude_code_settings_merges_user_environment():
    settings = build_claude_code_settings(
        {
            "env": {
                "API_TIMEOUT_MS": "300000",
                "ANTHROPIC_BASE_URL": "https://should-be-overridden.example.com",
            },
            "permissions": {"deny": ["WebFetch"]},
        },
        api_base="http://127.0.0.1:8000/v1",
        model="qwen3-coder",
        context_window=393216,
    )

    assert settings["env"]["API_TIMEOUT_MS"] == "300000"
    assert settings["env"]["ANTHROPIC_BASE_URL"] == "http://127.0.0.1:8000"
    assert settings["env"]["ANTHROPIC_API_KEY"] == "EMPTY"
    assert settings["env"]["ANTHROPIC_AUTH_TOKEN"] == "EMPTY"
    assert settings["env"]["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "qwen3-coder"
    assert settings["env"]["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "qwen3-coder"
    assert settings["env"]["ANTHROPIC_DEFAULT_HAIKU_MODEL"] == "qwen3-coder"
    assert settings["env"]["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "393216"
    assert settings["env"]["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"
    assert settings["env"]["CLAUDE_CODE_DISABLE_BACKGROUND_TASKS"] == "1"
    assert settings["env"]["DISABLE_AUTOUPDATER"] == "1"
    assert settings["permissions"] == {"deny": ["WebFetch"]}


def test_build_claude_code_settings_rejects_invalid_values():
    with pytest.raises(ValueError, match="context_window"):
        build_claude_code_settings({}, api_base="http://localhost:8000/v1", model="model", context_window=0)
    with pytest.raises(ValueError, match="env must be a dictionary"):
        build_claude_code_settings(
            {"env": []},
            api_base="http://localhost:8000/v1",
            model="model",
            context_window=262144,
        )


def test_build_claude_code_settings_applies_runtime_effort():
    settings = build_claude_code_settings(
        {"env": {"CLAUDE_CODE_EFFORT_LEVEL": "high"}},
        api_base="http://localhost:8000/v1",
        model="qwen",
        context_window=262144,
        effort="xhigh",
    )

    assert settings["env"]["CLAUDE_CODE_EFFORT_LEVEL"] == "xhigh"
    assert settings["env"]["CLAUDE_CODE_ALWAYS_ENABLE_EFFORT"] == "1"


def test_build_claude_code_settings_rejects_invalid_effort():
    with pytest.raises(ValueError, match="Unsupported claude_code_effort"):
        build_claude_code_settings(
            {},
            api_base="http://localhost:8000/v1",
            model="qwen",
            context_window=262144,
            effort="extreme",
        )


def test_build_claude_code_settings_disables_thinking_and_suppresses_effort():
    settings = build_claude_code_settings(
        {
            "env": {
                "MAX_THINKING_TOKENS": "16000",
                "CLAUDE_CODE_EFFORT_LEVEL": "high",
                "CLAUDE_CODE_ALWAYS_ENABLE_EFFORT": "1",
            }
        },
        api_base="http://localhost:8000/v1",
        model="qwen",
        context_window=262144,
        effort="xhigh",
        disable_thinking=True,
    )

    assert settings["env"]["MAX_THINKING_TOKENS"] == "0"
    assert "CLAUDE_CODE_EFFORT_LEVEL" not in settings["env"]
    assert "CLAUDE_CODE_ALWAYS_ENABLE_EFFORT" not in settings["env"]


def test_transform_claude_code_request_disables_thinking_and_preserves_other_output_config():
    request = {
        "model": "qwen",
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high", "format": {"type": "json_schema"}},
        "chat_template_kwargs": {
            "custom_option": True,
            "enable_thinking": True,
            "reasoning_effort": "high",
        },
    }

    transformed = transform_claude_code_request(
        request,
        {
            "enable_thinking": False,
            "preserve_thinking": False,
            "reasoning_effort": "xhigh",
        },
    )

    assert transformed["chat_template_kwargs"] == {
        "custom_option": True,
        "enable_thinking": False,
        "preserve_thinking": False,
    }
    assert "thinking" not in transformed
    assert transformed["output_config"] == {"format": {"type": "json_schema"}}
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"]["effort"] == "high"


def test_transform_claude_code_request_preserves_thinking_when_enabled():
    request = {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "medium"},
    }

    transformed = transform_claude_code_request(request, {"enable_thinking": True})

    assert transformed["thinking"] == {"type": "adaptive"}
    assert transformed["output_config"]["effort"] == "medium"
    assert transformed["chat_template_kwargs"] == {"enable_thinking": True}
