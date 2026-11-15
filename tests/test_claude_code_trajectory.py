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

from nemo_skills.inference.eval.claude_code_trajectory import convert_claude_code_stream_to_atif


def test_convert_claude_code_stream_to_atif():
    events = [
        {
            "type": "system",
            "subtype": "init",
            "session_id": "session-1",
            "model": "qwen3-coder",
        },
        {
            "type": "assistant",
            "session_id": "session-1",
            "parent_tool_use_id": None,
            "message": {
                "content": [
                    {"type": "thinking", "thinking": "Inspect the repository."},
                    {
                        "type": "tool_use",
                        "id": "tool-1",
                        "name": "Bash",
                        "input": {"command": "git status"},
                    },
                ],
                "usage": {
                    "input_tokens": 100,
                    "cache_read_input_tokens": 20,
                    "cache_creation_input_tokens": 5,
                    "output_tokens": 10,
                },
            },
        },
        {
            "type": "user",
            "session_id": "session-1",
            "parent_tool_use_id": None,
            "message": {
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool-1",
                        "content": "On branch main",
                    }
                ]
            },
        },
        {
            "type": "assistant",
            "session_id": "session-1",
            "parent_tool_use_id": None,
            "message": {
                "content": [{"type": "text", "text": "Implemented the fix."}],
                "usage": {"input_tokens": 120, "output_tokens": 8},
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "session_id": "session-1",
            "total_cost_usd": 0.25,
            "usage": {
                "input_tokens": 220,
                "cache_read_input_tokens": 20,
                "cache_creation_input_tokens": 5,
                "output_tokens": 18,
            },
        },
    ]

    trajectory = convert_claude_code_stream_to_atif(
        events,
        model_name="fallback-model",
        agent_version="2.1.259",
        initial_prompt="Fix the bug",
    )

    assert trajectory is not None
    assert trajectory["schema_version"] == "ATIF-v1.7"
    assert trajectory["session_id"] == "session-1"
    assert trajectory["agent"] == {
        "name": "claude-code",
        "version": "2.1.259",
        "model_name": "qwen3-coder",
    }
    assert trajectory["steps"][0] == {"step_id": 1, "source": "user", "message": "Fix the bug"}
    tool_step = trajectory["steps"][1]
    assert tool_step["reasoning_content"] == "Inspect the repository."
    assert tool_step["tool_calls"][0]["arguments"] == {"command": "git status"}
    assert tool_step["observation"]["results"][0]["content"] == "On branch main"
    assert tool_step["metrics"] == {
        "prompt_tokens": 125,
        "completion_tokens": 10,
        "cached_tokens": 20,
        "extra": {"cache_creation_tokens": 5},
    }
    assert trajectory["final_metrics"] == {
        "total_steps": 3,
        "total_prompt_tokens": 245,
        "total_completion_tokens": 18,
        "total_cached_tokens": 20,
        "total_cost_usd": 0.25,
        "extra": {"cache_creation_tokens": 5},
    }


def test_convert_claude_code_stream_ignores_subagent_messages():
    events = [
        {
            "type": "assistant",
            "session_id": "session-2",
            "parent_tool_use_id": "agent-tool",
            "message": {"content": [{"type": "text", "text": "subagent"}]},
        }
    ]

    assert convert_claude_code_stream_to_atif(events) is None
