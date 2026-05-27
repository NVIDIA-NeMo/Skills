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

"""Periodic table MCP tool for element and isotope data.

Wraps the ``periodictable`` library to provide element properties, isotope
data, and neutron scattering factors. No API key required.

Prerequisites:
    pip install periodictable

Usage:
    ++tool_modules=[nemo_skills.mcp.servers.chemistry.periodictable_tool::PeriodictableTool]
"""

import logging
from typing import Annotated, Any

from pydantic import Field

from nemo_skills.mcp.tool_manager import Tool

logger = logging.getLogger(__name__)


def _resolve_element(name_or_symbol: str):
    """Resolve an element from a name or symbol string."""
    import periodictable as pt

    s = name_or_symbol.strip()
    for el in pt.elements:
        if el.symbol == 0:
            continue
        if s.lower() == el.name.lower() or s.lower() == el.symbol.lower() or s == str(el.number):
            return el
    return None


def element_info(
    element: Annotated[str, Field(description="Element symbol, name, or atomic number (e.g. 'Fe', 'iron', '26').")],
) -> str:
    """Look up an element. Returns atomic mass, number, density, crystal structure, and isotope list."""
    el = _resolve_element(element)
    if el is None:
        return f"Element '{element}' not found. Try a symbol like 'Fe', name like 'iron', or number like '26'."

    lines = [f"**{el.name}** ({el.symbol})"]
    lines.append(f"Atomic number: {el.number}")
    lines.append(f"Atomic mass: {el.mass} u")
    if hasattr(el, "density") and el.density is not None:
        lines.append(f"Density: {el.density} g/cm^3")
    if hasattr(el, "crystal_structure") and el.crystal_structure is not None:
        lines.append(f"Crystal structure: {el.crystal_structure}")

    neutron = getattr(el, "neutron", None)
    if neutron is not None:
        if neutron.b_c is not None:
            lines.append(f"Neutron b_c: {neutron.b_c} fm")
        if neutron.coherent is not None:
            lines.append(f"Neutron coherent xs: {neutron.coherent} barn")
        if neutron.incoherent is not None:
            lines.append(f"Neutron incoherent xs: {neutron.incoherent} barn")
        if neutron.absorption is not None:
            lines.append(f"Neutron absorption xs: {neutron.absorption} barn")

    isotopes = [iso for iso in el if iso.abundance and iso.abundance > 0]
    if isotopes:
        lines.append("\nNatural isotopes:")
        for iso in sorted(isotopes, key=lambda x: -x.abundance):
            lines.append(f"  {el.symbol}-{iso.isotope}: mass={iso.mass:.6f} u, abundance={iso.abundance:.4f}%")

    return "\n".join(lines)


def isotope_info(
    element: Annotated[str, Field(description="Element symbol or name (e.g. 'U', 'uranium').")],
    mass_number: Annotated[int, Field(description="Mass number A of the isotope (e.g. 235 for U-235).")],
) -> str:
    """Look up a specific isotope. Returns mass, abundance, and neutron scattering data."""
    el = _resolve_element(element)
    if el is None:
        return f"Element '{element}' not found."

    try:
        iso = el[mass_number]
    except (KeyError, IndexError):
        return f"Isotope {el.symbol}-{mass_number} not found."

    lines = [f"**{el.symbol}-{mass_number}**"]
    if iso.mass is not None:
        lines.append(f"Mass: {iso.mass:.8f} u")
    if iso.abundance is not None:
        lines.append(f"Natural abundance: {iso.abundance:.4f}%")

    neutron = getattr(iso, "neutron", None)
    if neutron is not None:
        if neutron.b_c is not None:
            lines.append(f"Neutron b_c: {neutron.b_c} fm")
        if neutron.coherent is not None:
            lines.append(f"Neutron coherent xs: {neutron.coherent} barn")
        if neutron.incoherent is not None:
            lines.append(f"Neutron incoherent xs: {neutron.incoherent} barn")
        if neutron.absorption is not None:
            lines.append(f"Neutron absorption xs: {neutron.absorption} barn")

    return "\n".join(lines)


class PeriodictableTool(Tool):
    def __init__(self) -> None:
        self._config: dict[str, Any] = {}

    def default_config(self) -> dict[str, Any]:
        return dict(self._config)

    def configure(self, overrides: dict[str, Any] | None = None, context: dict[str, Any] | None = None) -> None:
        if overrides:
            self._config.update(overrides)

    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "element-info",
                "description": "Look up an element by symbol, name, or atomic number.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "element": {"type": "string", "description": "Element symbol, name, or atomic number."}
                    },
                    "required": ["element"],
                },
            },
            {
                "name": "isotope-info",
                "description": "Look up isotope mass, abundance, and neutron scattering data.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "element": {"type": "string", "description": "Element symbol or name."},
                        "mass_number": {"type": "integer", "description": "Mass number A of the isotope."},
                    },
                    "required": ["element", "mass_number"],
                },
            },
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any], extra_args: dict[str, Any] | None = None):
        arguments = dict(arguments or {})
        if tool_name == "element-info":
            return element_info(**arguments)
        if tool_name == "isotope-info":
            return isotope_info(**arguments)
        return f"Error: unknown tool '{tool_name}'"
