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

from nemo_skills.inference.eval.swebench import transform_mini_swe_agent_request


def test_transform_mini_swe_agent_request_adds_reasoning_alias_recursively():
    request = {
        "model": "model",
        "messages": [
            {
                "role": "assistant",
                "reasoning_content": "thinking",
                "metadata": {"reasoning_content": "nested"},
            },
            {
                "role": "assistant",
                "reasoning_content": "legacy",
                "reasoning": "authoritative",
            },
        ],
    }

    transformed = transform_mini_swe_agent_request(request)

    assert transformed["messages"][0]["reasoning"] == "thinking"
    assert transformed["messages"][0]["metadata"]["reasoning"] == "nested"
    assert transformed["messages"][1]["reasoning"] == "authoritative"
    assert "reasoning" not in request["messages"][0]


def test_transform_mini_swe_agent_request_promotes_litellm_provider_reasoning():
    request = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "provider_specific_fields": {
                    "reasoning": "thinking from LiteLLM",
                    "refusal": None,
                },
            },
            {
                "role": "assistant",
                "reasoning": "authoritative",
                "provider_specific_fields": {"reasoning": "provider fallback"},
            },
            {
                "role": "user",
                "provider_specific_fields": {"reasoning": "not assistant reasoning"},
            },
        ]
    }

    transformed = transform_mini_swe_agent_request(request)

    assert transformed["messages"][0]["reasoning"] == "thinking from LiteLLM"
    assert transformed["messages"][1]["reasoning"] == "authoritative"
    assert "reasoning" not in transformed["messages"][2]
    assert "reasoning" not in request["messages"][0]


def test_transform_mini_swe_agent_request_leaves_non_message_fields_unchanged():
    request = {
        "reasoning_content": "top-level",
        "messages": [{"role": "user", "content": "hello"}],
    }

    transformed = transform_mini_swe_agent_request(request)

    assert transformed == request
    assert transformed is not request
