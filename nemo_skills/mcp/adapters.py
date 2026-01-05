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
from omegaconf import DictConfig, OmegaConf

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


def load_schema_overrides(schema_overrides: dict | None) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Normalize schema overrides dict from Hydra/OmegaConf.

    Args:
        schema_overrides: Dict keyed by provider class name, then tool name, or None.
            Format: ProviderClassName -> tool_name -> (name, description, parameters)

    Returns:
        Normalized dict ready for use with format_tool_list_by_endpoint_type
    """
    if schema_overrides is None:
        return {}

    if isinstance(schema_overrides, DictConfig):
        schema_overrides = OmegaConf.to_container(schema_overrides, resolve=True)

    if not isinstance(schema_overrides, dict):
        raise ValueError(f"schema_overrides must be dict or None, got {type(schema_overrides)}")

    normalized = {}
    for provider_class, provider_overrides in schema_overrides.items():
        if not isinstance(provider_overrides, dict):
            raise ValueError(f"Override for provider '{provider_class}' must be a dict")

        normalized[provider_class] = {}
        for tool_name, cfg in provider_overrides.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"Override for tool '{tool_name}' in '{provider_class}' must be a dict")
            normalized[provider_class][tool_name] = {
                "name": cfg.get("name"),
                "description": cfg.get("description"),
                "parameters": cfg.get("parameters"),
            }

    return normalized


def apply_schema_overrides(
    tool: Dict[str, Any], override_config: Dict[str, Any] | None
) -> tuple[Dict[str, Any], Dict[str, str]]:
    """Apply schema overrides to a tool. Returns (transformed_tool, {new_param: orig_param})."""
    if not override_config:
        return tool, {}

    transformed = copy.deepcopy(tool)
    for key in ("name", "description"):
        if override_config.get(key) is not None:
            transformed[key] = override_config[key]

    param_overrides = override_config.get("parameters", {})
    if not param_overrides:
        return transformed, {}

    schema = transformed.get("input_schema", {})
    props, required = schema.get("properties", {}), set(schema.get("required", []))

    for name, cfg in param_overrides.items():
        if name not in props:
            raise ValueError(f"Parameter '{name}' not in schema")
        if not isinstance(cfg, dict):
            raise ValueError(f"Override for '{name}' must be a dict")

    new_props, new_required, mapping = {}, [], {}
    for orig, param in props.items():
        ovr = param_overrides.get(orig, {})
        new = ovr.get("name", orig)
        new_props[new] = {**param, **{k: v for k, v in ovr.items() if k != "name"}}
        if new != orig:
            mapping[new] = orig
        if orig in required:
            new_required.append(new)

    transformed["input_schema"] = {**schema, "properties": new_props, "required": new_required}
    return transformed, mapping


def remap_tool_call(tool_name: str, args: dict, mappings: dict) -> tuple[str, dict]:
    """Remap a tool call from model names back to original tool schema names."""
    original_tool = mappings.get("tool_names", {}).get(tool_name, tool_name)
    param_mapping = mappings.get("parameters", {}).get(tool_name, {})
    original_args = {param_mapping.get(k, k): v for k, v in args.items()}
    return original_tool, original_args


def format_tool_list_by_endpoint_type(
    tools, endpoint_type: EndpointType, schema_overrides: Dict[str, Dict[str, Dict[str, Any]]] | None = None
) -> tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """
    Format tool list for the given endpoint type, applying schema overrides.

    Returns:
        Tuple of (formatted_tools, mappings_dict) where mappings_dict has:
        - "tool_names": {model_name: original_name}
        - "parameters": {tool_name: {model_param: original_param}}
    """
    schema_overrides = schema_overrides or {}
    mappings = {"tool_names": {}, "parameters": {}}
    transformed_tools = []

    for tool in tools:
        original_name = tool["name"]
        provider = schema_overrides.get(tool.get("server")) or {}
        override = provider.get(original_name)

        transformed, param_mapping = apply_schema_overrides(tool, override)
        transformed_tools.append(transformed)

        new_name = transformed["name"]
        if new_name != original_name:
            mappings["tool_names"][new_name] = original_name
        if param_mapping:
            mappings["parameters"][new_name] = param_mapping

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

    return formatted, mappings


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
