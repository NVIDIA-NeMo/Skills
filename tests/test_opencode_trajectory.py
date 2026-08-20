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

from nemo_skills.inference.eval.opencode_trajectory import convert_opencode_session_to_atif


def test_convert_opencode_session_export_to_atif():
    session = {
        "info": {
            "id": "ses_test",
            "version": "1.17.11",
            "model": {"id": "Qwen3-Coder", "providerID": "nemo"},
            "cost": 0,
            "tokens": {
                "input": 100,
                "output": 25,
                "reasoning": 5,
                "cache": {"read": 20, "write": 4},
            },
        },
        "messages": [
            {
                "info": {"role": "user", "time": {"created": 1_700_000_000_000}},
                "parts": [{"type": "text", "text": "Fix the bug"}],
            },
            {
                "info": {
                    "role": "assistant",
                    "modelID": "Qwen3-Coder",
                    "providerID": "nemo",
                    "time": {"created": 1_700_000_001_000},
                },
                "parts": [
                    {"type": "reasoning", "text": "Inspect the code."},
                    {
                        "type": "tool",
                        "tool": "bash",
                        "callID": "call_1",
                        "state": {"input": {"command": "git status"}, "output": "clean"},
                    },
                    {"type": "text", "text": "Done"},
                    {
                        "type": "step-finish",
                        "cost": 0,
                        "tokens": {
                            "input": 100,
                            "output": 25,
                            "reasoning": 5,
                            "cache": {"read": 20, "write": 4},
                        },
                    },
                ],
            },
        ],
    }

    trajectory = convert_opencode_session_to_atif(session)

    assert trajectory is not None
    assert trajectory["schema_version"] == "ATIF-v1.7"
    assert trajectory["session_id"] == "ses_test"
    assert trajectory["agent"] == {
        "name": "opencode",
        "version": "1.17.11",
        "model_name": "nemo/Qwen3-Coder",
    }
    assert trajectory["steps"][0]["source"] == "user"
    assert trajectory["steps"][0]["message"] == "Fix the bug"
    assistant = trajectory["steps"][1]
    assert assistant["reasoning_content"] == "Inspect the code."
    assert assistant["tool_calls"][0]["arguments"] == {"command": "git status"}
    assert assistant["observation"]["results"][0]["content"] == "clean"
    assert assistant["metrics"]["prompt_tokens"] == 120
    assert trajectory["final_metrics"] == {
        "total_steps": 2,
        "total_prompt_tokens": 120,
        "total_completion_tokens": 25,
        "total_cached_tokens": 20,
        "extra": {"reasoning_tokens": 5, "cache_write_tokens": 4},
    }


def test_convert_opencode_session_uses_fallback_metadata_and_step_totals():
    session = {
        "info": {"id": "ses_fallback"},
        "messages": [
            {
                "info": {"role": "assistant"},
                "parts": [
                    {"type": "text", "text": "Answer"},
                    {
                        "type": "step-finish",
                        "tokens": {"input": 7, "output": 3, "cache": {"read": 0, "write": 0}},
                    },
                ],
            }
        ],
    }

    trajectory = convert_opencode_session_to_atif(
        session,
        model_name="/models/Qwen",
        agent_version="1.17.11",
    )

    assert trajectory is not None
    assert trajectory["agent"]["model_name"] == "/models/Qwen"
    assert trajectory["agent"]["version"] == "1.17.11"
    assert trajectory["final_metrics"]["total_prompt_tokens"] == 7
    assert trajectory["final_metrics"]["total_completion_tokens"] == 3
