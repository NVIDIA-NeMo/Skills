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
    ++tool_modules=[nemo_skills.mcp.servers.periodictable_tool::PeriodictableTool]
"""

import logging
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from nemo_skills.mcp.tool_providers import MCPClientTool

logger = logging.getLogger(__name__)

mcp = FastMCP(name="periodictable")


def _resolve_element(name_or_symbol: str):
    """Resolve an element from a name or symbol string."""
    import periodictable as pt

    s = name_or_symbol.strip()
    for el in pt.elements:
        if el.symbol == 0:
            continue
        if s.lower() == el.name.lower() or s == el.symbol or s == str(el.number):
            return el
    return None


@mcp.tool(name="element-info")
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

    try:
        if el.neutron.b_c is not None:
            lines.append(f"Neutron b_c: {el.neutron.b_c} fm")
        if el.neutron.coherent is not None:
            lines.append(f"Neutron coherent xs: {el.neutron.coherent} barn")
        if el.neutron.incoherent is not None:
            lines.append(f"Neutron incoherent xs: {el.neutron.incoherent} barn")
        if el.neutron.absorption is not None:
            lines.append(f"Neutron absorption xs: {el.neutron.absorption} barn")
    except AttributeError:
        pass

    isotopes = [iso for iso in el if iso.abundance and iso.abundance > 0]
    if isotopes:
        lines.append("\nNatural isotopes:")
        for iso in sorted(isotopes, key=lambda x: -x.abundance):
            lines.append(f"  {el.symbol}-{iso.isotope}: mass={iso.mass:.6f} u, abundance={iso.abundance:.4f}%")

    return "\n".join(lines)


@mcp.tool(name="isotope-info")
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

    try:
        n = iso.neutron
        if n.b_c is not None:
            lines.append(f"Neutron b_c: {n.b_c} fm")
        if n.coherent is not None:
            lines.append(f"Neutron coherent xs: {n.coherent} barn")
        if n.incoherent is not None:
            lines.append(f"Neutron incoherent xs: {n.incoherent} barn")
        if n.absorption is not None:
            lines.append(f"Neutron absorption xs: {n.absorption} barn")
    except AttributeError:
        pass

    return "\n".join(lines)


class PeriodictableTool(MCPClientTool):
    def __init__(self) -> None:
        super().__init__()
        self.apply_config_updates(
            {
                "client": "nemo_skills.mcp.clients.MCPStdioClient",
                "client_params": {
                    "command": "python",
                    "args": ["-m", "nemo_skills.mcp.servers.periodictable_tool"],
                },
            }
        )


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
