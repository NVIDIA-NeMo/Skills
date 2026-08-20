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

from nemo_skills.inference.eval.swebench import (
    OPENCODE_DEFAULT_OUTPUT_TOKEN_MAX,
    OPENCODE_PROVIDER_ID,
    build_opencode_config,
)


def test_build_opencode_config_points_at_local_openai_server():
    config = build_opencode_config(
        agent_config={
            "permission": {"bash": "allow"},
            "provider": {
                OPENCODE_PROVIDER_ID: {
                    "models": {
                        "/models/Qwen3-Coder": {"options": {"chat_template_kwargs": {"preserve_thinking": True}}}
                    }
                }
            },
        },
        api_base="http://10.0.0.1:5000/v1",
        model="/models/Qwen3-Coder",
        context_window=393216,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_effort": "xhigh",
            }
        },
        agent_max_turns=400,
        tokens_to_generate=4096,
    )

    provider = config["provider"][OPENCODE_PROVIDER_ID]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == "http://10.0.0.1:5000/v1"
    assert provider["options"]["apiKey"] == "EMPTY"
    assert provider["models"]["/models/Qwen3-Coder"]["id"] == "/models/Qwen3-Coder"
    assert provider["models"]["/models/Qwen3-Coder"]["temperature"] is True
    assert provider["models"]["/models/Qwen3-Coder"]["options"]["top_k"] == 20
    assert provider["models"]["/models/Qwen3-Coder"]["options"]["chat_template_kwargs"] == {
        "enable_thinking": True,
        "preserve_thinking": True,
        "reasoning_effort": "xhigh",
    }
    assert provider["models"]["/models/Qwen3-Coder"]["limit"]["context"] == 393216
    assert provider["models"]["/models/Qwen3-Coder"]["limit"]["output"] == 4096
    assert config["agent"]["build"]["temperature"] == 1.0
    assert config["agent"]["build"]["top_p"] == 0.95
    assert config["agent"]["build"]["steps"] == 400


def test_build_opencode_config_merges_user_keys():
    config = build_opencode_config(
        agent_config={
            "experimental": {"continue_loop_on_deny": True},
            "default_agent": "custom",
            "agent": {"custom": {"temperature": 0.2, "prompt": "Custom agent prompt"}},
        },
        api_base="http://127.0.0.1:8000/v1",
        model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        context_window=262144,
        temperature=0.7,
        top_p=0.8,
        top_k=None,
        extra_body={},
        agent_max_turns=250,
    )

    assert config["experimental"]["continue_loop_on_deny"] is True
    assert "Qwen/Qwen3-Coder-30B-A3B-Instruct" in config["provider"][OPENCODE_PROVIDER_ID]["models"]
    assert config["agent"]["custom"]["temperature"] == 0.7
    assert config["agent"]["custom"]["top_p"] == 0.8
    assert config["agent"]["custom"]["steps"] == 250
    assert config["agent"]["custom"]["prompt"] == "Custom agent prompt"
    model = config["provider"][OPENCODE_PROVIDER_ID]["models"]["Qwen/Qwen3-Coder-30B-A3B-Instruct"]
    assert model["limit"]["output"] == OPENCODE_DEFAULT_OUTPUT_TOKEN_MAX
    assert "options" not in model
