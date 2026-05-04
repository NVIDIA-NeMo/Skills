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
    ++tool_modules=[nemo_skills.mcp.servers.physics.coolprop_tool::CoolPropTool]
"""

import logging
from typing import Annotated, Any

from pydantic import Field

from nemo_skills.mcp.tool_manager import Tool

logger = logging.getLogger(__name__)

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


def fluid_list() -> str:
    """List all fluids available in CoolProp."""
    import CoolProp.CoolProp as CP

    fluids = sorted(CP.FluidsList())
    return f"**{len(fluids)} fluids available:**\n" + ", ".join(fluids)


class CoolPropTool(Tool):
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
                "name": "fluid-property",
                "description": "Calculate a thermophysical property of a fluid at temperature and pressure in SI units.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "fluid": {"type": "string", "description": "Fluid name, e.g. Water, Nitrogen, R134a, CO2."},
                        "output_property": {
                            "type": "string",
                            "description": "CoolProp output property code, e.g. D, C, H, S, V, L.",
                        },
                        "temperature": {"type": "number", "description": "Temperature in Kelvin."},
                        "pressure": {"type": "number", "description": "Pressure in Pascals."},
                    },
                    "required": ["fluid", "output_property", "temperature", "pressure"],
                },
            },
            {
                "name": "fluid-list",
                "description": "List fluids available in CoolProp.",
                "input_schema": {"type": "object", "properties": {}},
            },
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any], extra_args: dict[str, Any] | None = None):
        arguments = dict(arguments or {})
        if tool_name == "fluid-property":
            return fluid_property(**arguments)
        if tool_name == "fluid-list":
            return fluid_list()
        return f"Error: unknown tool '{tool_name}'"
