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

import httpx
from httpx import RemoteProtocolError
from mcp.server.fastmcp import FastMCP
from omegaconf import OmegaConf
from pydantic import Field

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.mcp.tool_manager import Tool
from nemo_skills.mcp.utils import add_config_args, load_mcp_config

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    output_dict: Dict[str, str]
    session_id: Any | None  # uuid


mcp = FastMCP(name="python_tool")

# Initialized from config in main()
sandbox = None

DEFAULT_EXEC_TIMEOUT_S = 10


def get_python_tool_description(exec_timeout_s: float = DEFAULT_EXEC_TIMEOUT_S) -> str:
    return (
        "Call this function to execute Python code in a stateful Jupyter notebook environment. "
        f"Python will respond with the output of the execution or time out after {exec_timeout_s:.1f} seconds."
    )


description = get_python_tool_description()


@mcp.tool(name="stateful_python_code_exec", description=description)
async def stateful_python_code_exec(
    code: Annotated[str, Field(description="Code to execute")],
    session_id: Annotated[str | None, Field(description="Session id for session persistence")] = None,
    timeout: Annotated[float, Field(description="Time in seconds to allow the job to run")] = DEFAULT_EXEC_TIMEOUT_S,
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


class DirectPythonTool(Tool):
    """Python code execution tool that calls the sandbox directly, bypassing MCP.

    This is the in-process implementation used by PythonTool. It eliminates the
    MCP protocol overhead (subprocess spawning, MCP session initialization,
    JSON-RPC serialization) by calling sandbox.execute_code() directly via HTTP.

    Config keys:
        - hide_args: controls which args are stripped from schemas and sanitized at runtime
        - exec_timeout_s: default execution timeout

    Usage:
        tool_modules=["nemo_skills.mcp.servers.python_tool::PythonTool"]
    """

    def __init__(self, exec_timeout_s: float = DEFAULT_EXEC_TIMEOUT_S) -> None:
        self._config: Dict[str, Any] = {
            # MCP-specific keys (client, client_params, init_hook) are intentionally absent here
            "hide_args": {"stateful_python_code_exec": ["session_id", "timeout"]},
            "exec_timeout_s": exec_timeout_s,
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
                "description": get_python_tool_description(self._config.get("exec_timeout_s", DEFAULT_EXEC_TIMEOUT_S)),
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
        timeout = extra_args.get("timeout", self._config.get("exec_timeout_s", DEFAULT_EXEC_TIMEOUT_S))
        session_id = self.requests_to_sessions[request_id] if request_id is not None else None

        # Validate required `code` argument. The MCP path gets this via Pydantic at the FastMCP
        # boundary; since we bypass FastMCP we need to validate ourselves so a malformed tool call
        # surfaces as a handleable error instead of crashing the tool.
        code = arguments.get("code")
        if not isinstance(code, str):
            output_dict = {
                "process_status": "fail",
                "stdout": "",
                "stderr": "Error: missing required argument 'code'",
            }
        else:
            try:
                output_dict, session_id = await self._sandbox.execute_code(
                    code,
                    language="ipython",
                    timeout=timeout,
                    session_id=session_id,
                )
            except (httpx.HTTPError, RemoteProtocolError):
                # Transport/protocol errors talking to the sandbox — log details, return generic.
                logger.exception("Sandbox communication error during stateful_python_code_exec")
                output_dict = {"process_status": "fail", "stdout": "", "stderr": "Error connecting to sandbox"}
                session_id = None
            except Exception:
                # Catch-all so a poorly-formed call or transient issue never crashes the RL run.
                # Log full details server-side but keep the model-facing stderr generic to avoid
                # leaking framework internals into what is supposed to be sandbox output.
                logger.exception("Unexpected error during stateful_python_code_exec")
                output_dict = {"process_status": "fail", "stdout": "", "stderr": "Error executing code"}
                session_id = None

        if request_id is not None:
            self.requests_to_sessions[request_id] = session_id

        output = f"{output_dict.get('stdout', '')}{output_dict.get('stderr', '')}"
        if output.endswith("\n"):
            output = output[:-1]
        return output

    async def shutdown(self) -> None:
        if self._sandbox is not None:
            session_ids = {
                str(session_id) for session_id in self.requests_to_sessions.values() if session_id is not None
            }
            for session_id in session_ids:
                try:
                    await self._sandbox.delete_session(session_id)
                except Exception:
                    logger.exception("Failed to delete sandbox session %s during shutdown", session_id)
            try:
                await self._sandbox.close()
            except Exception:
                logger.exception("Failed to close sandbox HTTP session during shutdown")
        self.requests_to_sessions.clear()

    async def cleanup_request(self, request_id: str) -> None:
        session_id = self.requests_to_sessions.get(request_id)
        if session_id is None:
            return
        if self._sandbox is not None:
            try:
                await self._sandbox.delete_session(str(session_id))
            except Exception:
                logger.exception("Failed to delete sandbox session %s during cleanup_request", session_id)
        self.requests_to_sessions.pop(request_id, None)


class PythonTool(DirectPythonTool):
    """Default Python tool implementation.

    Uses the direct in-process provider (DirectPythonTool) instead of stdio MCP
    transport so generation jobs do not depend on a subprocess teardown path.
    Kept as a subclass for backward compatibility with existing
    ``tool_modules=[...python_tool::PythonTool]`` references.
    """

    pass


if __name__ == "__main__":
    main()
