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

from litellm.types.utils import ChatCompletionMessageToolCall

from nemo_skills.inference.model.base import EndpointType


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


def format_tool_list_by_endpoint_type(tools, endpoint_type: EndpointType):
    if endpoint_type == EndpointType.chat:
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
    elif endpoint_type == EndpointType.responses:
        return [
            {
                "type": "function",
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
                "strict": True,  # Less vllm errors through structured output
            }
            for t in tools
        ]
    else:
        raise ValueError(f"Unsupported completion type for tool list: {endpoint_type}")


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


def format_tool_response_by_endpoint_type(tool_call, result, endpoint_type: EndpointType):
    if endpoint_type == EndpointType.chat:
        return {
            "role": "tool",
            "name": tool_call["function"]["name"],
            "tool_call_id": tool_call["id"],
            "content": json.dumps(result) if not isinstance(result, str) else result,
        }
    elif endpoint_type == EndpointType.responses:
        return {
            "type": "function_call_output",
            "call_id": tool_call["call_id"],
            "output": json.dumps(result) if not isinstance(result, str) else result,
        }
    else:
        raise ValueError(f"Unsupported completion type for tool call: {endpoint_type}")


def get_tool_details_by_endpoint_type(tool_call, endpoint_type: EndpointType):
    if endpoint_type == EndpointType.chat:
        tool_name = tool_call["function"]["name"]
        tool_args = tool_call["function"]["arguments"]
    elif endpoint_type == EndpointType.responses:
        assert tool_call["type"] == "function_call", "Tool call must be a function call"
        tool_name = tool_call["name"]
        tool_args = tool_call["arguments"]
    else:
        raise ValueError(f"Unsupported completion type for tool call: {endpoint_type}")
    return tool_name, tool_args
