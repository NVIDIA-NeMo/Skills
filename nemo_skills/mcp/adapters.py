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

import copy
import json
from abc import ABC, abstractmethod
from typing import Any, Dict

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


def apply_schema_overrides(
    tool: Dict[str, Any], override_config: Dict[str, Any] | None
) -> tuple[Dict[str, Any], Dict[str, str]]:
    """
    Apply schema overrides to a single tool and build parameter mapping.

    Args:
        tool: Original tool dict with keys: name, description, input_schema
        override_config: Override configuration dict with optional keys:
            - name: Override tool name
            - description: Override tool description
            - parameters: Dict mapping new parameter names to parameter configs

    Returns:
        Tuple of (transformed_tool_dict, parameter_mapping)
        where parameter_mapping is {new_param_name: original_param_name}
    """
    if override_config is None:
        return tool, {}

    transformed = copy.deepcopy(tool)
    parameter_mapping = {}

    # Override tool name
    if override_config.get("name") is not None:
        transformed["name"] = override_config["name"]

    # Override tool description
    if override_config.get("description") is not None:
        transformed["description"] = override_config["description"]

    # Override parameters
    if override_config.get("parameters") is not None:
        original_schema = transformed.get("input_schema", {})
        original_properties = original_schema.get("properties", {})
        original_required = original_schema.get("required", [])
        original_param_names = list(original_properties.keys())

        # Build new properties dict and parameter mapping
        new_properties = {}
        new_required = []

        # Map new parameter names to original parameter names
        override_params = override_config["parameters"]
        override_param_list = list(override_params.items())

        for idx, (new_param_name, param_config) in enumerate(override_param_list):
            # Find the original parameter name
            # 1. Check for explicit "original_name" in param_config
            # 2. If new_param_name exists in original, use it (no rename, just override properties)
            # 3. If only one param in each, map 1:1
            # 4. Otherwise, match by position
            original_param_name = None
            if isinstance(param_config, dict):
                original_param_name = param_config.get("original_name")

            if original_param_name is None:
                # Check if new_param_name exists in original (no rename, just property override)
                if new_param_name in original_properties:
                    original_param_name = new_param_name
                elif len(override_param_list) == 1 and len(original_param_names) == 1:
                    # Single parameter: map 1:1
                    original_param_name = original_param_names[0]
                elif idx < len(original_param_names):
                    # Match by position
                    original_param_name = original_param_names[idx]
                else:
                    # New parameter not in original, or can't determine mapping
                    # Assume it's a new parameter (no mapping needed)
                    original_param_name = None

            # Build new parameter schema
            if original_param_name and original_param_name in original_properties:
                original_param = original_properties[original_param_name]
                new_param = copy.deepcopy(original_param)
                # Override with new config (excluding original_name)
                if isinstance(param_config, dict):
                    new_param.update({k: v for k, v in param_config.items() if k != "original_name"})
                else:
                    # If param_config is not a dict, it might be just the type
                    new_param["type"] = param_config

                # Track mapping if names differ
                if new_param_name != original_param_name:
                    parameter_mapping[new_param_name] = original_param_name

                # Preserve required status
                if original_param_name in original_required:
                    new_required.append(new_param_name)
            else:
                # New parameter not in original - use param_config directly
                if isinstance(param_config, dict):
                    new_param = copy.deepcopy(param_config)
                    new_param.pop("original_name", None)
                else:
                    new_param = {"type": param_config}
                # New parameters are not required unless explicitly specified
                if isinstance(param_config, dict) and param_config.get("required", False):
                    new_required.append(new_param_name)

            new_properties[new_param_name] = new_param

        # Add any original parameters not overridden
        # Track which original params were handled (either renamed or updated in place)
        handled_original_params = set(parameter_mapping.values())
        # Also track params that were updated in place (same name)
        for new_name, orig_name in parameter_mapping.items():
            handled_original_params.add(orig_name)
        # Also check if any override params have the same name as original (updated in place)
        for new_param_name in override_params.keys():
            if new_param_name in original_properties:
                handled_original_params.add(new_param_name)

        for orig_param_name, orig_param in original_properties.items():
            if orig_param_name not in handled_original_params:
                new_properties[orig_param_name] = copy.deepcopy(orig_param)
                if orig_param_name in original_required:
                    new_required.append(orig_param_name)

        # Update input_schema
        transformed["input_schema"] = {
            **original_schema,
            "properties": new_properties,
            "required": new_required,
        }

    return transformed, parameter_mapping


def format_tool_list_by_endpoint_type(
    tools, endpoint_type: EndpointType, schema_overrides: Dict[str, Dict[str, Dict[str, Any]]] | None = None
) -> tuple[list[Dict[str, Any]], Dict[str, Dict[str, str]]]:
    """
    Format tool list for the given endpoint type, optionally applying schema overrides.

    Args:
        tools: List of tool dicts with keys: name, description, input_schema, server
        endpoint_type: The endpoint type (chat or responses)
        schema_overrides: Optional dict keyed by provider class name, then tool name.
            Format: {ProviderClassName: {tool_name: {name, description, parameters}}}

    Returns:
        Tuple of (formatted_tools_list, parameter_mapping_dict)
        where parameter_mapping_dict is {tool_name: {new_param: original_param}}
    """
    schema_overrides = schema_overrides or {}
    parameter_mapping = {}

    # Apply schema overrides and collect parameter mappings
    transformed_tools = []
    for tool in tools:
        provider_class = tool.get("server")  # Provider class name (e.g., "PythonTool")
        original_tool_name = tool["name"]

        # Look up override: first by provider class, then by tool name
        override_config = None
        if provider_class and provider_class in schema_overrides:
            provider_overrides = schema_overrides[provider_class]
            override_config = provider_overrides.get(original_tool_name)

        transformed_tool, tool_param_mapping = apply_schema_overrides(tool, override_config)
        transformed_tools.append(transformed_tool)

        # Store parameter mapping keyed by the NEW tool name (after override)
        new_tool_name = transformed_tool["name"]
        if tool_param_mapping:
            parameter_mapping[new_tool_name] = tool_param_mapping

    # Format for endpoint type
    if endpoint_type == EndpointType.chat:
        formatted = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["input_schema"],
                },
            }
            for t in transformed_tools
        ]
    elif endpoint_type == EndpointType.responses:
        formatted = [
            {
                "type": "function",
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
                "strict": True,  # Less vllm errors through structured output
            }
            for t in transformed_tools
        ]
    else:
        raise ValueError(f"Unsupported completion type for tool list: {endpoint_type}")

    return formatted, parameter_mapping


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
