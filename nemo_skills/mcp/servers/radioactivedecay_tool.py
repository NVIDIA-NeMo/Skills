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
    ++tool_modules=[nemo_skills.mcp.servers.radioactivedecay_tool::RadioactivedecayTool]
"""

import logging
import math
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from nemo_skills.mcp.tool_providers import MCPClientTool

logger = logging.getLogger(__name__)

mcp = FastMCP(name="radioactivedecay")

VALID_TIME_UNITS = {"ps", "ns", "us", "ms", "s", "m", "h", "d", "y", "ky", "My", "Gy", "Ty"}


@mcp.tool(name="nuclide-info")
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


@mcp.tool(name="decay-chain")
def decay_chain(
    nuclide: Annotated[str, Field(description="Starting nuclide (e.g. 'U-238', 'Co-60').")],
    time: Annotated[float, Field(description="Elapsed time for decay calculation.")],
    time_unit: Annotated[str, Field(description="Time unit: s, m, h, d, y, ky, My, Gy, Ty, ps, ns, us, ms.")] = "s",
) -> str:
    """Calculate the decay chain products and activities after a given time."""
    import radioactivedecay as rd

    if time_unit not in VALID_TIME_UNITS:
        return f"Invalid time unit '{time_unit}'. Valid units: {', '.join(sorted(VALID_TIME_UNITS))}"

    if not math.isfinite(time):
        return "Time must be a finite number."
    if time < 0:
        return "Time must be non-negative."

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


class RadioactivedecayTool(MCPClientTool):
    def __init__(self) -> None:
        super().__init__()
        self.apply_config_updates(
            {
                "client": "nemo_skills.mcp.clients.MCPStdioClient",
                "client_params": {
                    "command": "python",
                    "args": ["-m", "nemo_skills.mcp.servers.radioactivedecay_tool"],
                },
                "hide_args": {
                    "decay-chain": ["time_unit"],
                },
            }
        )


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
