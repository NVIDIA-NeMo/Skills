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
from collections import defaultdict
from dataclasses import dataclass
from typing import Annotated, Any, Dict, List

from httpx import RemoteProtocolError
from mcp.server.fastmcp import FastMCP
from omegaconf import OmegaConf
from pydantic import Field

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.mcp.tool_manager import Tool
from nemo_skills.mcp.tool_providers import MCPClientTool
from nemo_skills.mcp.utils import add_config_args, load_mcp_config

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    output_dict: Dict[str, str]
    session_id: Any | None  # uuid


mcp = FastMCP(name="python_tool")

# Initialized from config in main()
sandbox = None

# TODO: how should we control timeout in description?
description = (
    "Call this function to execute Python code in a stateful Jupyter notebook environment. "
    "Python will respond with the output of the execution or time out after 120.0 seconds."
)


@mcp.tool(name="stateful_python_code_exec", description=description)
async def stateful_python_code_exec(
    code: Annotated[str, Field(description="Code to execute")],
    session_id: Annotated[str | None, Field(description="Session id for session persistence")] = None,
    timeout: Annotated[float, Field(description="Time in seconds to allow the job to run")] = 10,
) -> ExecutionResult:
    language = "ipython"
    try:
        output_dict, session_id = await sandbox.execute_code(
            code, language=language, timeout=timeout, session_id=session_id
        )
    except RemoteProtocolError:
        output_dict = {"process_status": "fail", "stdout": "", "stderr": "Error connecting to sandbox"}
        session_id = None

    return {"output_dict": output_dict, "session_id": session_id}


def main():
    parser = argparse.ArgumentParser(description="MCP server for executing Python code in a sandbox")
    parser.add_argument(
        "--disable-session-restore",
        action="store_true",
        default=False,
        help="Skip replaying session history after sandbox worker restarts (overrides config)",
    )
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
    if args.disable_session_restore:
        sandbox_cfg["disable_session_restore"] = True

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
                "hide_args": {"stateful_python_code_exec": ["session_id", "timeout"]},
                # use explicit Hydra connector built from full context by default
                "init_hook": "hydra",
                # execution-specific default
                "exec_timeout_s": 10,
                "sandbox": {},
            }
        )
        self.requests_to_sessions = defaultdict(lambda: None)
        self._sandbox = None

    def configure(self, overrides: Dict[str, Any] | None = None, context: Dict[str, Any] | None = None) -> None:
        super().configure(overrides=overrides, context=context)

        sandbox_cfg = dict((context or {}).get("sandbox", {}))
        sandbox_cfg.update(self._config.get("sandbox", {}))
        sandbox_cfg.pop("sandbox_type", None)
        sandbox_type = (context or {}).get("sandbox", {}).get("sandbox_type", "local")
        sandbox_type = self._config.get("sandbox", {}).get("sandbox_type", sandbox_type)
        self._sandbox = get_sandbox(sandbox_type=sandbox_type, **sandbox_cfg)

    async def execute(self, tool_name: str, arguments: Dict[str, Any], extra_args: Dict[str, Any] | None = None):
        # Ensure timeout is sent via extra_args (post-sanitize), not in main arguments
        arguments = dict(arguments)
        # TODO: error handling?
        request_id = extra_args.pop("request_id")
        merged_extra = dict(extra_args or {})
        merged_extra.setdefault("timeout", self._config.get("exec_timeout_s", 10))
        merged_extra["session_id"] = self.requests_to_sessions[request_id]
        result = await self._client.call_tool(tool=tool_name, args=arguments, extra_args=merged_extra)
        self.requests_to_sessions[request_id] = result["session_id"]
        output = f"{result['output_dict']['stdout']}{result['output_dict']['stderr']}"
        if output.endswith("\n"):  # there is always a trailing newline, removing it
            output = output[:-1]
        return output

    async def shutdown(self) -> None:
        session_ids = {str(session_id) for session_id in self.requests_to_sessions.values() if session_id is not None}
        if self._sandbox is not None:
            for session_id in session_ids:
                await self._sandbox.delete_session(session_id)
            await self._sandbox.close()
        self.requests_to_sessions.clear()

    async def cleanup_request(self, request_id: str) -> None:
        session_id = self.requests_to_sessions.get(request_id)
        if session_id is None:
            return
        if self._sandbox is not None:
            await self._sandbox.delete_session(str(session_id))
        self.requests_to_sessions.pop(request_id, None)


class DirectPythonTool(Tool):
    """Python code execution tool that calls the sandbox directly, bypassing MCP.

    This is a drop-in replacement for PythonTool that eliminates the MCP protocol
    overhead (subprocess spawning, MCP session initialization, JSON-RPC serialization)
    by calling sandbox.execute_code() directly via HTTP.

    Shared config keys with PythonTool (so switching is just changing the module spec):
        - hide_args: controls which args are stripped from schemas and sanitized at runtime
        - exec_timeout_s: default execution timeout

    Usage:
        tool_modules=["nemo_skills.mcp.servers.python_tool::DirectPythonTool"]
    """

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {
            # Same keys/defaults as PythonTool (minus MCP-specific: client, client_params, init_hook)
            "hide_args": {"stateful_python_code_exec": ["session_id", "timeout"]},
            "exec_timeout_s": 10,
            "sandbox": {},
        }
        self._sandbox = None
        self._sanitize_keys: Dict[str, set] = {}
        self.requests_to_sessions: Dict[str, Any] = defaultdict(lambda: None)

    def default_config(self) -> Dict[str, Any]:
        return dict(self._config)

    def configure(self, overrides: Dict[str, Any] | None = None, context: Dict[str, Any] | None = None) -> None:
        if overrides:
            self._config.update(overrides)

        # Build sanitize sets from hide_args (same source of truth as MCP path)
        hide_args = self._config.get("hide_args", {})
        self._sanitize_keys = {tool: set(keys) for tool, keys in hide_args.items()}

        # Build sandbox config from context (same source as the MCP server's main())
        sandbox_cfg = dict((context or {}).get("sandbox", {}))
        sandbox_cfg.update(self._config.get("sandbox", {}))
        sandbox_cfg.pop("sandbox_type", None)
        sandbox_type = (context or {}).get("sandbox", {}).get("sandbox_type", "local")
        sandbox_type = self._config.get("sandbox", {}).get("sandbox_type", sandbox_type)
        self._sandbox = get_sandbox(sandbox_type=sandbox_type, **sandbox_cfg)

    async def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "stateful_python_code_exec",
                "description": description,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string", "description": "Code to execute"},
                    },
                    "required": ["code"],
                },
            }
        ]

    async def execute(
        self, tool_name: str, arguments: Dict[str, Any], extra_args: Dict[str, Any] | None = None
    ) -> str:
        # Strip model-supplied hidden args using hide_args config (same source as MCP sanitize())
        hidden = self._sanitize_keys.get(tool_name, set())
        arguments = {k: v for k, v in arguments.items() if k not in hidden}

        extra_args = dict(extra_args or {})
        request_id = extra_args.pop("request_id", None)
        timeout = extra_args.get("timeout", self._config.get("exec_timeout_s", 10))
        session_id = self.requests_to_sessions[request_id] if request_id is not None else None

        try:
            output_dict, session_id = await self._sandbox.execute_code(
                arguments["code"],
                language="ipython",
                timeout=timeout,
                session_id=session_id,
            )
        except RemoteProtocolError:
            output_dict = {"process_status": "fail", "stdout": "", "stderr": "Error connecting to sandbox"}
            session_id = None

        if request_id is not None:
            self.requests_to_sessions[request_id] = session_id

        output = f"{output_dict['stdout']}{output_dict['stderr']}"
        if output.endswith("\n"):
            output = output[:-1]
        return output

    async def shutdown(self) -> None:
        if self._sandbox is not None:
            session_ids = {
                str(session_id) for session_id in self.requests_to_sessions.values() if session_id is not None
            }
            for session_id in session_ids:
                await self._sandbox.delete_session(session_id)
            await self._sandbox.close()
        self.requests_to_sessions.clear()

    async def cleanup_request(self, request_id: str) -> None:
        session_id = self.requests_to_sessions.get(request_id)
        if session_id is None:
            return
        if self._sandbox is not None:
            await self._sandbox.delete_session(str(session_id))
        self.requests_to_sessions.pop(request_id, None)


if __name__ == "__main__":
    main()
