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

import argparse
import logging
from dataclasses import dataclass
from typing import Annotated, Any, Dict

from httpx import RemoteProtocolError
from mcp.server.fastmcp import FastMCP
from omegaconf import OmegaConf
from pydantic import Field

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.mcp.tool_providers import MCPClientTool
from nemo_skills.mcp.utils import add_config_args, load_mcp_config

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    process_status: str
    stderr: str
    stdout: str


mcp = FastMCP(name="python_tool")

# Initialized from config in main()
sandbox = None


@mcp.tool()
async def execute(
    code: Annotated[str, Field(description="Code to run in python interpreter")],
    session_id: Annotated[str | None, Field(description="Session id for session persistence")] = None,
    timeout: Annotated[float, Field(description="Time in seconds to allow the job to run")] = 10,
) -> ExecutionResult:
    """Executes the given python code"""
    language = "ipython"
    try:
        output, _ = await sandbox.execute_code(code, language=language, timeout=timeout, session_id=session_id)
    except RemoteProtocolError:
        return {"process_status": "fail", "stdout": "", "stderr": "Error connecting to sandbox"}
    return output


def main():
    parser = argparse.ArgumentParser(description="MCP server for executing Python code in a sandbox")
    add_config_args(parser)
    args = parser.parse_args()

    try:
        cfg = load_mcp_config(
            config=args.config,
            config_dir=args.config_dir,
            config_name=args.config_name,
        )
    except ValueError as e:
        logger.warning(f"{e} Falling back to default local sandbox config.")
        cfg = OmegaConf.create({"sandbox": {"sandbox_type": "local"}})

    global sandbox
    sandbox_cfg = OmegaConf.to_container(cfg.sandbox, resolve=True)
    sandbox = get_sandbox(**sandbox_cfg)
    # Initialize and run the server
    mcp.run(transport="stdio")


# ==============================
# Module-based tool implementation
# ==============================


class PythonTool(MCPClientTool):
    def __init__(self) -> None:
        super().__init__()
        # Defaults for stdio Python MCP using explicit client class
        self.apply_config_updates(
            {
                "client": "nemo_skills.mcp.clients.MCPStdioClient",
                "client_params": {
                    "command": "python",
                    "args": ["-m", "nemo_skills.mcp.servers.python_tool"],
                },
                # hide args from schemas and sanitize at runtime
                "hide_args": {"execute": ["session_id", "timeout"]},
                # use explicit Hydra connector built from full context by default
                "init_hook": "hydra",
                # execution-specific default
                "exec_timeout_s": 10,
            }
        )

    async def execute(self, tool_name: str, arguments: Dict[str, Any], extra_args: Dict[str, Any] | None = None):
        # Ensure timeout is sent via extra_args (post-sanitize), not in main arguments
        arguments = dict(arguments)
        merged_extra = dict(extra_args or {})
        merged_extra.setdefault("timeout", self._config.get("exec_timeout_s", 10))
        return await self._client.call_tool(tool=tool_name, args=arguments, extra_args=merged_extra)

    async def shutdown(self) -> None:
        return None


if __name__ == "__main__":
    main()
