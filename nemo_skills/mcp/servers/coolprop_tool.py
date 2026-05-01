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

"""CoolProp MCP tool for thermophysical fluid properties.

Wraps the ``CoolProp`` library to look up density, viscosity, conductivity,
specific heat, and other properties for 124 fluids. No API key required.

Prerequisites:
    pip install CoolProp

Usage:
    ++tool_modules=[nemo_skills.mcp.servers.coolprop_tool::CoolPropTool]
"""

import logging
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from nemo_skills.mcp.tool_providers import MCPClientTool

logger = logging.getLogger(__name__)

mcp = FastMCP(name="coolprop")

PROPERTY_DESCRIPTIONS = {
    "D": "Density [kg/m^3]",
    "H": "Specific enthalpy [J/kg]",
    "S": "Specific entropy [J/(kg*K)]",
    "C": "Specific heat at constant pressure Cp [J/(kg*K)]",
    "CVMASS": "Specific heat at constant volume Cv [J/(kg*K)]",
    "V": "Dynamic viscosity [Pa*s]",
    "L": "Thermal conductivity [W/(m*K)]",
    "P": "Pressure [Pa]",
    "T": "Temperature [K]",
    "Q": "Vapor quality [-]",
    "SPEED_OF_SOUND": "Speed of sound [m/s]",
    "SURFACE_TENSION": "Surface tension [N/m]",
    "PRANDTL": "Prandtl number [-]",
    "ISENTROPIC_EXPANSION_COEFFICIENT": "Isentropic expansion coefficient [-]",
}


@mcp.tool(name="fluid-property")
def fluid_property(
    fluid: Annotated[str, Field(description="Fluid name (e.g. 'Water', 'Nitrogen', 'R134a', 'CO2').")],
    output_property: Annotated[
        str,
        Field(
            description=(
                "Property to calculate. Common codes: "
                "D (density), C (Cp), CVMASS (Cv), H (enthalpy), S (entropy), "
                "V (viscosity), L (conductivity), SPEED_OF_SOUND, PRANDTL."
            )
        ),
    ],
    temperature: Annotated[float, Field(description="Temperature in Kelvin.")],
    pressure: Annotated[float, Field(description="Pressure in Pascals.")],
) -> str:
    """Calculate a thermophysical property of a fluid at given temperature and pressure (SI units)."""
    import CoolProp.CoolProp as CP

    if temperature <= 0:
        return "Temperature must be positive (in Kelvin)."
    if pressure <= 0:
        return "Pressure must be positive (in Pascals)."

    try:
        value = CP.PropsSI(output_property, "T", temperature, "P", pressure, fluid)
    except ValueError as e:
        return f"CoolProp error for {fluid}: {e}"

    desc = PROPERTY_DESCRIPTIONS.get(output_property, output_property)
    return f"**{fluid}** at T={temperature} K, P={pressure} Pa\n{desc}: {value:.6g}"


@mcp.tool(name="fluid-list")
def fluid_list() -> str:
    """List all fluids available in CoolProp."""
    import CoolProp.CoolProp as CP

    fluids = sorted(CP.FluidsList())
    return f"**{len(fluids)} fluids available:**\n" + ", ".join(fluids)


class CoolPropTool(MCPClientTool):
    def __init__(self) -> None:
        super().__init__()
        self.apply_config_updates(
            {
                "client": "nemo_skills.mcp.clients.MCPStdioClient",
                "client_params": {
                    "command": "python",
                    "args": ["-m", "nemo_skills.mcp.servers.coolprop_tool"],
                },
            }
        )


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
