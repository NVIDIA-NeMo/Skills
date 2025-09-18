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


import json
from abc import ABC, abstractmethod
from typing import List

from litellm.types.utils import ChatCompletionMessageToolCall


# ==============================
# ADAPTER INTERFACES
# ==============================
class ToolSchemaAdapter(ABC):
    @abstractmethod
    def convert(self, tools: list[dict]) -> list[dict]:
        """Convert MCP tool definitions into model-specific schema."""
        raise NotImplementedError("Subclasses must implement this method.")


class ToolCallInterpreter(ABC):
    @abstractmethod
    def parse(self, raw_call: dict) -> dict:
        raise NotImplementedError("Subclasses must implement this method.")


class ToolResponseFormatter(ABC):
    @abstractmethod
    def format(self, tool_call: ChatCompletionMessageToolCall, result: dict) -> dict:
        """Format the response from a tool call."""
        raise NotImplementedError("Subclasses must implement this method.")


# ==============================
# ADAPTER IMPLEMENTATIONS
# ==============================


class OpenAISchemaAdapter(ToolSchemaAdapter):
    # https://platform.openai.com/docs/guides/function-calling#defining-functions
    def convert(self, tools):
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]


class OpenAICallInterpreter(ToolCallInterpreter):
    def parse(self, tool_call):
        fn = tool_call.function
        tool = fn.name
        return {"tool_name": tool, "args": json.loads(fn.arguments)}


class CompletionResponseFormatter(ToolResponseFormatter):
    # https://qwen.readthedocs.io/en/latest/framework/function_call.html#id2
    def format(self, tool_call: ChatCompletionMessageToolCall, result):
        return {
            "role": "tool",
            "content": json.dumps(result),
            "tool_call_id": tool_call.id,
        }


class ChatCompletionCallInterpreter(ToolCallInterpreter):
    """Convert tool calls to a chat message item.

    Should be broadly compatible with any OpenAI-like APIs,
    and HuggingFace Chat templates.

    NOTE(sanyamk): For error handling, delay JSON parsing of arguments to the model class.
    """

    def parse(self, tool_calls: List[ChatCompletionMessageToolCall]):
        tool_calls = [
            {
                "type": tool_call.type,
                "id": tool_call.id,
                "function": {"name": tool_call.function.name, "arguments": tool_call.function.arguments},
            }
            for tool_call in tool_calls
        ]

        return {"role": "assistant", "tool_calls": tool_calls}


class ChatCompletionResponseFormatter(ToolResponseFormatter):
    """Convert tool call result to chat message item.

    Use in conjunction with `ChatCompletionCallInterpreter`.
    """

    def format(self, tool_call, result):
        return {
            "role": "tool",
            "name": tool_call["function"]["name"],
            "tool_call_id": tool_call["id"],
            "content": json.dumps(result) if not isinstance(result, str) else result,
        }
