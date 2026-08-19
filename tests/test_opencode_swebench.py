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

from nemo_skills.inference.eval.swebench import OPENCODE_PROVIDER_ID, build_opencode_config


def test_build_opencode_config_points_at_local_openai_server():
    config = build_opencode_config(
        agent_config={"permission": {"bash": "allow"}},
        api_base="http://10.0.0.1:5000/v1",
        model="/models/Qwen3-Coder",
        tokens_to_generate=4096,
    )

    provider = config["provider"][OPENCODE_PROVIDER_ID]
    assert provider["npm"] == "@ai-sdk/openai-compatible"
    assert provider["options"]["baseURL"] == "http://10.0.0.1:5000/v1"
    assert provider["options"]["apiKey"] == "EMPTY"
    assert provider["models"]["/models/Qwen3-Coder"]["id"] == "/models/Qwen3-Coder"
    assert provider["models"]["/models/Qwen3-Coder"]["limit"]["output"] == 4096


def test_build_opencode_config_merges_user_keys():
    config = build_opencode_config(
        agent_config={"experimental": {"continue_loop_on_deny": True}},
        api_base="http://127.0.0.1:8000/v1",
        model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
    )

    assert config["experimental"]["continue_loop_on_deny"] is True
    assert "Qwen/Qwen3-Coder-30B-A3B-Instruct" in config["provider"][OPENCODE_PROVIDER_ID]["models"]
