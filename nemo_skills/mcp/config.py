# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

"""Unified MCP configuration: schema (nested dataclasses) and loader utilities.

This merges the prior config_schema and config_loader modules into one place.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import field
from typing import Any

from omegaconf import DictConfig

from nemo_skills.mcp.utils import locate
from nemo_skills.utils import nested_dataclass

# ==============================
# Schema (nested dataclasses)
# ==============================


@nested_dataclass(kw_only=True)
class MCPAdaptersConfig:
    schema_adapter: str
    call_interpreter: str
    response_formatter: str


@nested_dataclass(kw_only=True)
class MCPClientParamsBase:
    hide_args: dict[str, list[str]] = field(default_factory=dict)
    disabled_tools: list[str] = field(default_factory=list)
    enabled_tools: list[str] = field(default_factory=list)
    output_formatter: Any | None = None
    init_hook: Any | None = None


@nested_dataclass(kw_only=True)
class MCPStdioClientParams(MCPClientParamsBase):
    command: str
    args: list[str] = field(default_factory=list)


@nested_dataclass(kw_only=True)
class MCPStreamableHttpClientParams(MCPClientParamsBase):
    base_url: str


@nested_dataclass(kw_only=True)
class MCPToolConfig:
    id: str
    client: str
    params: dict[str, Any] = field(default_factory=dict)


@nested_dataclass(kw_only=True)
class MCPConfig:
    adapters: MCPAdaptersConfig
    tools: list[MCPToolConfig] = field(default_factory=list)


# ==============================
# Loader utilities
# ==============================

RESOLVABLE_PARAM_KEYS = {"output_formatter", "init_hook"}


def _is_locate_mapping(value: Any) -> bool:
    """Return True if value is a mapping-like object containing a "$locate" key.

    Supports both plain dict and OmegaConf DictConfig.
    """
    try:
        return isinstance(value, Mapping) and ("$locate" in value)
    except Exception:
        return False


def _resolve_special(value: Any, full_cfg: DictConfig) -> Any:
    if isinstance(value, str) and value == "@@full_config":
        return full_cfg
    return value


def _resolve_locate_mapping(spec: Mapping, full_cfg: DictConfig) -> Any:
    target = locate(spec.get("$locate"))
    raw_args = spec.get("args", [])
    raw_kwargs = spec.get("kwargs", {})
    args = [resolve_value(a, full_cfg) for a in raw_args]
    kwargs = {k: resolve_value(v, full_cfg) for k, v in raw_kwargs.items()}
    return target(*args, **kwargs)


def resolve_value(value: Any, full_cfg: DictConfig) -> Any:
    if _is_locate_mapping(value):
        return _resolve_locate_mapping(value, full_cfg)
    return _resolve_special(value, full_cfg)


def resolve_adapters(cfg: DictConfig):
    adapters_cfg = cfg.adapters
    schema_adapter_obj = locate(adapters_cfg.schema_adapter)
    call_interpreter_obj = locate(adapters_cfg.call_interpreter)
    response_formatter_obj = locate(adapters_cfg.response_formatter)

    schema_adapter = schema_adapter_obj() if isinstance(schema_adapter_obj, type) else schema_adapter_obj
    call_interpreter = call_interpreter_obj() if isinstance(call_interpreter_obj, type) else call_interpreter_obj
    response_formatter = (
        response_formatter_obj() if isinstance(response_formatter_obj, type) else response_formatter_obj
    )
    return schema_adapter, call_interpreter, response_formatter
