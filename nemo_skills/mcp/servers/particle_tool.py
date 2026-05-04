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

"""Particle physics MCP tool wrapping the PDG particle database.

Provides lookup of particle properties (mass, width, charge, spin, lifetime)
and search by name. Uses the ``particle`` pip package. No API key required.

Prerequisites:
    pip install particle

Usage:
    ++tool_modules=[nemo_skills.mcp.servers.particle_tool::ParticleTool]
"""

import logging
from typing import Annotated, Any

from pydantic import Field

from nemo_skills.mcp.tool_manager import Tool

logger = logging.getLogger(__name__)

MAX_SEARCH_RESULTS = 10


def _format_particle(p) -> str:
    lines = [f"**{p.name}**"]
    lines.append(f"PDG ID: {p.pdgid}")
    if p.latex_name:
        lines.append(f"LaTeX: {p.latex_name}")
    if p.mass is not None:
        lines.append(f"Mass: {p.mass} MeV/c^2")
    if p.width is not None:
        lines.append(f"Width: {p.width} MeV")
    if p.charge is not None:
        lines.append(f"Charge: {p.charge} e")
    if p.J is not None:
        lines.append(f"Spin (J): {p.J}")
    if p.lifetime is not None:
        lines.append(f"Lifetime: {p.lifetime:.6e} ns")
    if p.anti_flag:
        lines.append(f"Anti-flag: {p.anti_flag.name}")
    if p.P is not None:
        lines.append(f"Parity (P): {'+' if p.P == 1 else '-'}")
    if p.C is not None:
        lines.append(f"C-parity: {'+' if p.C == 1 else '-'}")
    return "\n".join(lines)


def particle_lookup(
    name_or_id: Annotated[
        str,
        Field(description="Particle name (e.g. 'pi+', 'K0', 'J/psi(1S)') or PDG ID as a string (e.g. '211')."),
    ],
) -> str:
    """Look up a particle by name or PDG ID. Returns mass, width, charge, spin, lifetime, and quantum numbers."""
    from particle import InvalidParticle, Particle, ParticleNotFound

    try:
        p = Particle.from_pdgid(int(name_or_id))
        return _format_particle(p)
    except (ValueError, ParticleNotFound, InvalidParticle):
        pass

    try:
        p = Particle.from_name(name_or_id)
        return _format_particle(p)
    except (ParticleNotFound, InvalidParticle):
        pass

    matches = Particle.findall(name_or_id)
    if matches:
        return _format_particle(matches[0])

    return f"Particle '{name_or_id}' not found. Try a standard name like 'pi+', 'K-', 'D0', or a PDG ID."


def particle_search(
    query: Annotated[
        str, Field(description="Search query - substring match on particle names (e.g. 'K', 'charm', 'omega').")
    ],
) -> str:
    """Search for particles by name. Returns a list of matching particles with key properties."""
    from particle import Particle

    matches = Particle.findall(query)
    if not matches:
        return f"No particles found matching '{query}'."

    matches = matches[:MAX_SEARCH_RESULTS]
    results = []
    for p in matches:
        mass_str = f"{p.mass} MeV/c^2" if p.mass is not None else "n/a"
        results.append(f"**{p.name}** (PDG {p.pdgid}) - mass: {mass_str}, charge: {p.charge}")
    header = f"Found {len(results)} particle(s) matching '{query}':\n"
    return header + "\n".join(results)


class ParticleTool(Tool):
    def __init__(self) -> None:
        self._config: dict[str, Any] = {}

    def default_config(self) -> dict[str, Any]:
        return dict(self._config)

    def configure(self, overrides: dict[str, Any] | None = None, context: dict[str, Any] | None = None) -> None:
        if overrides:
            unknown = ", ".join(sorted(overrides))
            raise ValueError(f"ParticleTool does not support overrides: {unknown}")

    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "particle-lookup",
                "description": "Look up a particle by name or PDG ID.",
                "input_schema": {
                    "type": "object",
                    "properties": {"name_or_id": {"type": "string", "description": "Particle name or PDG ID."}},
                    "required": ["name_or_id"],
                },
            },
            {
                "name": "particle-search",
                "description": "Search for particles by name.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Substring query on particle names."}},
                    "required": ["query"],
                },
            },
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any], extra_args: dict[str, Any] | None = None):
        if extra_args:
            unknown = ", ".join(sorted(map(str, extra_args)))
            raise ValueError(f"ParticleTool does not support extra_args: {unknown}")
        arguments = dict(arguments or {})
        if tool_name == "particle-lookup":
            return particle_lookup(**arguments)
        if tool_name == "particle-search":
            return particle_search(**arguments)
        return f"Error: unknown tool '{tool_name}'"
