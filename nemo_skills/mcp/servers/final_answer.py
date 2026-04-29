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

from typing import Any, Dict, List

from nemo_skills.mcp.tool_manager import Tool

FINAL_ANSWER_TOOL_NAME = "final_answer"

description = "Use this tool to provide the final answer to the user's question."


class FinalAnswerTool(Tool):
    """Final answer tool that bypasses MCP and returns the supplied answer."""

    def default_config(self) -> Dict[str, Any]:
        return {}

    def configure(self, overrides: Dict[str, Any] | None = None, context: Dict[str, Any] | None = None) -> None:
        return None

    async def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": FINAL_ANSWER_TOOL_NAME,
                "description": description,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The final answer to the user's question.",
                        },
                    },
                    "required": ["answer"],
                },
            }
        ]

    async def execute(
        self, tool_name: str, arguments: Dict[str, Any], extra_args: Dict[str, Any] | None = None
    ) -> str:
        if tool_name != FINAL_ANSWER_TOOL_NAME:
            raise ValueError(f"Unsupported tool: {tool_name}")
        return f"\\boxed{{{arguments['answer']}}}"
