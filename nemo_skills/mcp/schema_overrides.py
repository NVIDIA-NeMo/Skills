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

from typing import Any, Dict


def load_schema_overrides(schema_overrides: dict | None) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Normalize schema overrides dict.

    Args:
        schema_overrides: Dict keyed by provider class name, then tool name, or None.
            Format: {ProviderClassName: {tool_name: {name, description, parameters}}}
            Hydra handles file loading (e.g., ++schema_overrides=@file.yaml),
            so we only need to accept the resulting dict.

    Returns:
        Dict keyed by provider class name, then tool name with override configurations

    Raises:
        ValueError: If structure is malformed
    """
    if schema_overrides is None:
        return {}

    if not isinstance(schema_overrides, dict):
        raise ValueError(f"schema_overrides must be dict or None, got {type(schema_overrides)}")

    return _normalize_overrides(schema_overrides)


def _normalize_overrides(overrides: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Normalize override structure and validate.

    Args:
        overrides: Raw override dict keyed by provider class name

    Returns:
        Normalized override dict keyed by provider class name, then tool name
    """
    normalized = {}
    for provider_class, provider_overrides in overrides.items():
        if not isinstance(provider_overrides, dict):
            raise ValueError(
                f"Override for provider '{provider_class}' must be a dict, got {type(provider_overrides)}"
            )

        normalized[provider_class] = {}
        for tool_name, override_config in provider_overrides.items():
            if not isinstance(override_config, dict):
                raise ValueError(
                    f"Override for tool '{tool_name}' in provider '{provider_class}' must be a dict, "
                    f"got {type(override_config)}"
                )

            normalized[provider_class][tool_name] = {
                "name": override_config.get("name"),
                "description": override_config.get("description"),
                "parameters": override_config.get("parameters"),
            }

    return normalized
