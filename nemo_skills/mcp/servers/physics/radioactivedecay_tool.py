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

"""Radioactive decay MCP tool for nuclear decay chain calculations.

Wraps the ``radioactivedecay`` library to provide nuclide information and
time-evolution of decay chains. No API key required.

Prerequisites:
    pip install radioactivedecay

Usage:
    ++tool_modules=[nemo_skills.mcp.servers.physics.radioactivedecay_tool::RadioactivedecayTool]
"""

import logging
import math
from typing import Annotated, Any

from pydantic import Field

from nemo_skills.mcp.tool_manager import Tool

logger = logging.getLogger(__name__)

VALID_TIME_UNITS = {"ps", "ns", "us", "ms", "s", "m", "h", "d", "y", "ky", "My", "Gy", "Ty"}


def nuclide_info(
    nuclide: Annotated[str, Field(description="Nuclide in standard notation (e.g. 'H-3', 'U-238', 'Co-60').")],
) -> str:
    """Look up a radioactive nuclide. Returns half-life, decay modes, progeny, and branching fractions."""
    import radioactivedecay as rd

    try:
        nuc = rd.Nuclide(nuclide)
    except ValueError:
        return f"Nuclide '{nuclide}' not found. Use notation like 'H-3', 'U-238', 'Co-60'."

    lines = [f"**{nuc.nuclide}**"]
    lines.append(f"Atomic number (Z): {nuc.Z}")
    lines.append(f"Mass number (A): {nuc.A}")

    half_life_s = nuc.half_life()
    if half_life_s == float("inf"):
        lines.append("Half-life: stable")
    else:
        for unit in ["y", "d", "h", "m", "s"]:
            hl = nuc.half_life(unit)
            if hl >= 1.0:
                lines.append(f"Half-life: {hl:.6g} {unit}")
                break
        else:
            lines.append(f"Half-life: {half_life_s:.6g} s")

    modes = nuc.decay_modes()
    if modes:
        lines.append(f"Decay modes: {', '.join(str(m) for m in modes)}")

    progeny = nuc.progeny()
    branching = nuc.branching_fractions()
    if progeny:
        lines.append("Progeny:")
        for daughter, bf in zip(progeny, branching):
            lines.append(f"  {daughter} (branching fraction: {bf:.6g})")

    return "\n".join(lines)


def decay_chain(
    nuclide: Annotated[str, Field(description="Starting nuclide (e.g. 'U-238', 'Co-60').")],
    time: Annotated[float, Field(description="Elapsed time for decay calculation.")],
    time_unit: Annotated[str, Field(description="Time unit: s, m, h, d, y, ky, My, Gy, Ty, ps, ns, us, ms.")] = "s",
) -> str:
    """Calculate the decay chain products and activities after a given time."""
    if time_unit not in VALID_TIME_UNITS:
        return f"Invalid time unit '{time_unit}'. Valid units: {', '.join(sorted(VALID_TIME_UNITS))}"

    if not math.isfinite(time):
        return "Time must be a finite number."
    if time < 0:
        return "Time must be non-negative."

    import radioactivedecay as rd

    try:
        rd.Nuclide(nuclide)
    except ValueError:
        return f"Nuclide '{nuclide}' not found. Use notation like 'H-3', 'U-238', 'Co-60'."

    inv = rd.Inventory({nuclide: 1.0}, "Bq")
    decayed = inv.decay(time, time_unit)
    activities = decayed.activities("Bq")

    lines = [f"**Decay of {nuclide} after {time} {time_unit}**", ""]
    lines.append(f"{'Nuclide':<12} {'Activity (Bq)':>15}")
    lines.append("-" * 28)
    for nuc_name, activity in sorted(activities.items(), key=lambda x: -x[1]):
        if activity > 1e-15:
            lines.append(f"{str(nuc_name):<12} {activity:>15.6e}")

    return "\n".join(lines)


class RadioactivedecayTool(Tool):
    def __init__(self) -> None:
        self._config: dict[str, Any] = {"time_unit": "s"}

    def default_config(self) -> dict[str, Any]:
        return dict(self._config)

    def configure(self, overrides: dict[str, Any] | None = None, context: dict[str, Any] | None = None) -> None:
        if not overrides:
            return

        allowed = {"time_unit"}
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(f"Unsupported RadioactivedecayTool override(s): {sorted(unknown)}")

        time_unit = overrides.get("time_unit", self._config["time_unit"])
        if time_unit not in VALID_TIME_UNITS:
            raise ValueError(f"Invalid time unit '{time_unit}'. Valid units: {', '.join(sorted(VALID_TIME_UNITS))}")
        self._config["time_unit"] = time_unit

    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "nuclide-info",
                "description": "Look up half-life, decay modes, progeny, and branching fractions for a nuclide.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "nuclide": {"type": "string", "description": "Nuclide notation, e.g. H-3, U-238, Co-60."}
                    },
                    "required": ["nuclide"],
                },
            },
            {
                "name": "decay-chain",
                "description": "Calculate decay-chain product activities after an elapsed time.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "nuclide": {"type": "string", "description": "Starting nuclide."},
                        "time": {"type": "number", "description": "Elapsed time for the decay calculation."},
                    },
                    "required": ["nuclide", "time"],
                },
            },
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any], extra_args: dict[str, Any] | None = None):
        arguments = dict(arguments or {})
        if tool_name == "nuclide-info":
            return nuclide_info(**arguments)
        if tool_name == "decay-chain":
            arguments.setdefault("time_unit", self._config["time_unit"])
            return decay_chain(**arguments)
        return f"Error: unknown tool '{tool_name}'"
