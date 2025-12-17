# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import types

import pytest

# Dummy client to exercise MCPClientMeta behavior without real I/O
from nemo_skills.mcp.clients import MCPClient, MCPStdioClient, MCPStreamableHttpClient
from nemo_skills.mcp.tool_manager import Tool, ToolManager


class DummyClient(MCPClient):
    def __init__(self):
        # Pre-populate with a simple tool list; will also be returned by list_tools()
        self.tools = [
            {
                "name": "execute",
                "description": "Run code",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "code": {"type": "string"},
                        "session_id": {"type": "string"},
                        "timeout": {"type": "integer"},
                    },
                    "required": ["code", "session_id"],
                },
            },
            {
                "name": "echo",
                "description": "Echo input",
                "input_schema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        ]

    async def list_tools(self):
        return list(self.tools)

    async def call_tool(self, tool: str, args: dict):
        # Enforce allowed/disabled rules like real clients do
        self._assert_tool_allowed(tool)
        if tool == "execute":
            return {"ran": True, "code": args.get("code")}
        if tool == "echo":
            return {"echo": args.get("text")}
        return {"unknown": tool, "args": args}


class MinimalClient(MCPClient):
    # No __init__; tests default attribute injection via metaclass __call__
    async def list_tools(self):
        return []

    async def call_tool(self, tool: str, args: dict):
        return {"ok": True}


@pytest.mark.asyncio
async def test_metaclass_list_tools_hides_and_filters():
    client = DummyClient(
        hide_args={"execute": ["session_id", "timeout"]},
        disabled_tools=["echo"],
    )
    tools = await client.list_tools()

    # Only "execute" should remain due to disabled_tools
    names = {t["name"] for t in tools}
    assert names == {"execute"}

    execute = tools[0]
    schema = execute["input_schema"]
    assert "session_id" not in schema["properties"]
    assert "timeout" not in schema["properties"]
    assert "code" in schema["properties"]
    # required should be updated (removed hidden keys)
    assert "session_id" not in schema.get("required", [])


@pytest.mark.asyncio
async def test_metaclass_enabled_tools_allowlist_and_missing_check():
    # When enabled_tools is non-empty: only those are returned, and missing raises
    client = DummyClient(enabled_tools=["execute"])  # allow only execute
    tools = await client.list_tools()
    assert [t["name"] for t in tools] == ["execute"]

    client_missing = DummyClient(enabled_tools=["execute", "missing_tool"])  # missing
    with pytest.raises(ValueError):
        await client_missing.list_tools()


@pytest.mark.asyncio
async def test_metaclass_call_tool_output_formatter_and_init_hook():
    hook_called = {"flag": False}

    def init_hook(self):
        hook_called["flag"] = True
        setattr(self, "_ready", True)

    def formatter(result):
        # Convert results to a simple string signature
        if isinstance(result, dict) and "ran" in result:
            return f"ran:{result.get('code')}"
        return str(result)

    client = DummyClient(output_formatter=formatter, init_hook=init_hook)
    assert hook_called["flag"] is True
    assert getattr(client, "_ready", False) is True

    out = await client.call_tool("execute", {"code": "print(1)"})
    assert out == "ran:print(1)"


def test_minimal_client_defaults_and_sanitize():
    # Minimal client with no __init__ still gets default attributes
    c = MinimalClient()
    assert hasattr(c, "_hide_args") and c._hide_args == {}
    assert hasattr(c, "_enabled_tools") and isinstance(c._enabled_tools, set)
    assert hasattr(c, "_disabled_tools") and isinstance(c._disabled_tools, set)

    # Sanitize removes hidden keys
    c._hide_args = {"tool": ["secret", "token"]}
    clean = c.sanitize("tool", {"x": 1, "secret": 2, "token": 3})
    assert clean == {"x": 1}


@pytest.mark.asyncio
async def test_stdio_env_inheritance_with_minimal_server(monkeypatch, tmp_path):
    # Ensure parent env has sentinel
    monkeypatch.setenv("TEST_ENV_PROP", "sentinel_value")

    # Write a minimal stdio MCP server script that echoes env back
    server_code = (
        "import os\n"
        "from dataclasses import dataclass\n"
        "from typing import Annotated\n"
        "from mcp.server.fastmcp import FastMCP\n"
        "from pydantic import Field\n"
        "\n"
        "@dataclass\n"
        "class EnvResult:\n"
        "    value: str | None\n"
        "\n"
        "mcp = FastMCP(name='env_echo_tool')\n"
        "\n"
        "@mcp.tool()\n"
        "async def echo_env(var_name: Annotated[str, Field(description='Environment variable name to read')]) -> EnvResult:\n"
        "    return {'value': os.environ.get(var_name)}\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    mcp.run(transport='stdio')\n"
    )
    script_path = tmp_path / "env_echo_tool_tmp.py"
    script_path.write_text(server_code)

    # Launch the temporary stdio server via MCP client
    client = MCPStdioClient(command="python", args=[str(script_path)])

    # Call tool to read env var from server process
    result = await client.call_tool("echo_env", {"var_name": "TEST_ENV_PROP"})

    assert isinstance(result, dict)
    # Structured content passthrough returns dict with value
    assert result.get("value") == "sentinel_value"


class DummyTool(Tool):
    def __init__(self) -> None:
        self._cfg = {}

    def default_config(self):
        return {}

    def configure(self, overrides=None, context=None):
        return None

    async def list_tools(self):
        return [
            {
                "name": "execute",
                "description": "Run code",
                "input_schema": {
                    "type": "object",
                    "properties": {"code": {"type": "string"}},
                    "required": ["code"],
                },
            },
            {
                "name": "echo",
                "description": "Echo input",
                "input_schema": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        ]

    async def execute(self, tool_name: str, arguments: dict, extra_args: dict | None = None):
        if tool_name == "execute":
            return {"ran": True, "code": arguments.get("code")}
        if tool_name == "echo":
            return {"echo": arguments.get("text")}
        return {"unknown": tool_name, "args": arguments}


# Helper class for test_tool_manager_cache_and_duplicate_detection
# Defined at module level so it can be imported via locate()
class CountingTool(DummyTool):
    # Use a class variable that's mutable to track calls
    # This will be shared across all instances
    call_count = 0

    def __init__(self) -> None:
        super().__init__()

    async def list_tools(self):
        # Increment the class variable
        CountingTool.call_count += 1
        return await super().list_tools()


# Helper class for duplicate tool detection test
class DupTool(DummyTool):
    async def list_tools(self):
        lst = await super().list_tools()
        return [lst[0], lst[0]]  # duplicate names within same tool


@pytest.mark.asyncio
async def test_tool_manager_list_and_execute_with_class_locator():
    # Register this test module's DummyTool via module locator
    # Use __name__ to get actual module path (works in both local and CI)
    tm = ToolManager(module_specs=[f"{__name__}::DummyTool"], overrides={}, context={})
    tools = await tm.list_all_tools(use_cache=False)
    names = sorted(t["name"] for t in tools)
    assert names == ["echo", "execute"]

    result = await tm.execute_tool("execute", {"code": "x=1"})
    assert result == {"ran": True, "code": "x=1"}


@pytest.mark.asyncio
async def test_tool_manager_cache_and_duplicate_detection():
    import sys

    # Reset counter before test - access via sys.modules to ensure we get the right class
    this_module = sys.modules[__name__]
    CountingToolClass = getattr(this_module, "CountingTool")
    CountingToolClass.call_count = 0

    # Use __name__ to get the actual module path (works in both local and CI environments)
    module_path = __name__
    tm = ToolManager(module_specs=[f"{module_path}::CountingTool"], overrides={}, context={})
    _ = await tm.list_all_tools(use_cache=True)
    _ = await tm.list_all_tools(use_cache=True)
    assert CountingToolClass.call_count == 1, f"Expected 1 call, got {CountingToolClass.call_count}"
    with pytest.raises(ValueError) as excinfo:
        _ = await tm.list_all_tools(use_cache=False)
    assert "Duplicate raw tool name across providers: 'execute'" in str(excinfo.value)
    assert CountingToolClass.call_count == 2, f"Expected 2 calls, got {CountingToolClass.call_count}"

    tm2 = ToolManager(module_specs=[f"{module_path}::DupTool"], overrides={}, context={})
    tools2 = await tm2.list_all_tools(use_cache=False)
    names2 = sorted(t["name"] for t in tools2)
    assert names2 == ["execute"]


@pytest.mark.asyncio
async def test_stdio_client_list_tools_hide_and_call_tool_with_output_formatter(monkeypatch):
    # Build fakes
    class ToolObj:
        def __init__(self, name, description, input_schema=None, inputSchema=None):
            self.name = name
            self.description = description
            if input_schema is not None:
                self.input_schema = input_schema
            if inputSchema is not None:
                self.inputSchema = inputSchema

    class ToolsResp:
        def __init__(self, tools):
            self.tools = tools

    class ResultObj:
        def __init__(self, structured):
            self.structuredContent = structured

    class FakeSession:
        def __init__(self, *_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def initialize(self):
            return None

        async def list_tools(self):
            return ToolsResp(
                [
                    ToolObj(
                        name="execute",
                        description="Run",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "code": {"type": "string"},
                                "session_id": {"type": "string"},
                                "timeout": {"type": "integer"},
                            },
                            "required": ["code", "session_id"],
                        },
                    ),
                    ToolObj(
                        name="echo",
                        description="Echo",
                        inputSchema={
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    ),
                ]
            )

        async def call_tool(self, tool, arguments):
            return ResultObj({"tool": tool, "args": arguments})

    class FakeStdioCtx:
        async def __aenter__(self):
            return ("r", "w")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    import nemo_skills.mcp.clients as clients_mod

    monkeypatch.setattr(clients_mod, "ClientSession", FakeSession)
    monkeypatch.setattr(clients_mod, "stdio_client", lambda *_: FakeStdioCtx())

    formatted = []

    def output_formatter(result):
        formatted.append(result)
        return {"formatted": True, "data": result}

    client = MCPStdioClient(
        command="python",
        args=["-m", "nemo_skills.mcp.servers.python_tool"],
        hide_args={"execute": ["session_id", "timeout"]},
        enabled_tools=["execute", "echo"],
        output_formatter=output_formatter,
    )

    tools = await client.list_tools()
    # Ensure hide_args pruned and names preserved
    names = sorted(t["name"] for t in tools)
    assert names == ["echo", "execute"]
    exec_tool = next(t for t in tools if t["name"] == "execute")
    props = exec_tool["input_schema"]["properties"]
    assert "session_id" not in props and "timeout" not in props and "code" in props

    # call_tool should enforce allowlist and apply output formatter
    out = await client.call_tool("execute", {"code": "print(1)"})
    assert out == {"formatted": True, "data": {"tool": "execute", "args": {"code": "print(1)"}}}
    # formatter received the pre-formatted structured content
    assert formatted and formatted[-1] == {"tool": "execute", "args": {"code": "print(1)"}}


@pytest.mark.asyncio
async def test_stdio_client_enabled_tools_enforcement(monkeypatch):
    class FakeSession:
        def __init__(self, *_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def initialize(self):
            return None

        async def list_tools(self):
            # Minimal list
            class T:
                def __init__(self):
                    self.name = "execute"
                    self.description = "d"
                    self.input_schema = {"type": "object"}

            class R:
                def __init__(self, tools):
                    self.tools = tools

            return R([T()])

        async def call_tool(self, tool, arguments):
            class Res:
                def __init__(self, content):
                    self.structuredContent = content

            return Res({"ok": True})

    class FakeStdioCtx:
        async def __aenter__(self):
            return ("r", "w")

        async def __aexit__(self, exc_type, exc, tb):
            return False

    import nemo_skills.mcp.clients as clients_mod

    monkeypatch.setattr(clients_mod, "ClientSession", FakeSession)
    monkeypatch.setattr(clients_mod, "stdio_client", lambda *_: FakeStdioCtx())

    client = MCPStdioClient(command="python", enabled_tools=["only_this_tool"])  # allowlist excludes "execute"
    with pytest.raises(PermissionError):
        await client.call_tool("execute", {})


@pytest.mark.asyncio
async def test_streamable_http_client_list_and_call_tool(monkeypatch):
    class ToolObj:
        def __init__(self, name, description, input_schema=None, inputSchema=None):
            self.name = name
            self.description = description
            if input_schema is not None:
                self.input_schema = input_schema
            if inputSchema is not None:
                self.inputSchema = inputSchema

    class ToolsResp:
        def __init__(self, tools):
            self.tools = tools

    class ResultObj:
        def __init__(self, structured=None):
            self.structuredContent = structured

    class FakeSession:
        def __init__(self, *_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def initialize(self):
            return None

        async def list_tools(self):
            return ToolsResp(
                [
                    ToolObj("t1", "desc", input_schema={"type": "object"}),
                    ToolObj("t2", "desc", inputSchema={"type": "object"}),
                ]
            )

        async def call_tool(self, tool, arguments):
            if tool == "t1":
                return ResultObj({"ok": True})
            # No structured content and no text content -> client should return error dict
            return types.SimpleNamespace(structuredContent=None, content=None)

    class FakeHttpCtx:
        async def __aenter__(self):
            return ("r", "w", None)

        async def __aexit__(self, exc_type, exc, tb):
            return False

    import nemo_skills.mcp.clients as clients_mod

    monkeypatch.setattr(clients_mod, "ClientSession", FakeSession)
    monkeypatch.setattr(clients_mod, "streamablehttp_client", lambda *_: FakeHttpCtx())

    client = MCPStreamableHttpClient(base_url="https://example.com/mcp")
    tools = await client.list_tools()
    assert sorted(t["name"] for t in tools) == ["t1", "t2"]

    # structured content present -> return structured
    out1 = await client.call_tool("t1", {})
    assert out1 == {"ok": True}

    # structured content absent and no text content -> return error dict (not raw object)
    out2 = await client.call_tool("t2", {"x": 1})
    assert out2 == {"error": "No content returned from tool"}


@pytest.mark.asyncio
async def test_streamable_http_client_enforcement(monkeypatch):
    class FakeSession:
        def __init__(self, *_):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def initialize(self):
            return None

        async def list_tools(self):
            class T:
                def __init__(self):
                    self.name = "t1"
                    self.description = "d"
                    self.input_schema = {"type": "object"}

            class R:
                def __init__(self, tools):
                    self.tools = tools

            return R([T()])

        async def call_tool(self, tool, arguments):
            return types.SimpleNamespace(structuredContent=None)

    class FakeHttpCtx:
        async def __aenter__(self):
            return ("r", "w", None)

        async def __aexit__(self, exc_type, exc, tb):
            return False

    import nemo_skills.mcp.clients as clients_mod

    monkeypatch.setattr(clients_mod, "ClientSession", FakeSession)
    monkeypatch.setattr(clients_mod, "streamablehttp_client", lambda *_: FakeHttpCtx())

    client = MCPStreamableHttpClient(base_url="https://example.com/mcp", enabled_tools=["only_t2"])  # not including t1
    with pytest.raises(PermissionError):
        await client.call_tool("t1", {})


@pytest.mark.asyncio
async def test_tool_manager_with_schema_overrides():
    """Test ToolManager integration with schema overrides."""
    from nemo_skills.inference.model.base import EndpointType
    from nemo_skills.mcp.adapters import format_tool_list_by_endpoint_type, load_schema_overrides

    tm = ToolManager(module_specs=[f"{__name__}::DummyTool"], overrides={}, context={})
    tools = await tm.list_all_tools(use_cache=False)

    schema_overrides = {
        "DummyTool": {
            "execute": {
                "name": "renamed_execute",
                "parameters": {"code": {"name": "script"}},  # rename 'code' -> 'script' for model
            }
        }
    }
    loaded_overrides = load_schema_overrides(schema_overrides)
    formatted_tools, mappings = format_tool_list_by_endpoint_type(
        tools, EndpointType.chat, schema_overrides=loaded_overrides
    )

    renamed_tool = next((t for t in formatted_tools if t["function"]["name"] == "renamed_execute"), None)
    assert renamed_tool is not None
    assert "script" in renamed_tool["function"]["parameters"]["properties"]
    assert "code" not in renamed_tool["function"]["parameters"]["properties"]
    assert mappings["parameters"]["renamed_execute"] == {"script": "code"}
    assert mappings["tool_names"]["renamed_execute"] == "execute"


def test_schema_override_nonexistent_param_fails():
    """Overriding a parameter that doesn't exist in the schema must fail early.

    This also covers the hidden-arg case: when hide_args removes a param from the
    schema before overrides are applied, attempting to override that (now-missing)
    param will trigger the same error.
    """
    from nemo_skills.mcp.adapters import apply_schema_overrides

    tool = {
        "name": "test",
        "description": "Test",
        "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}, "required": []},
    }
    # Try to override 'script' which doesn't exist (tool only has 'code')
    with pytest.raises(ValueError, match="Parameter 'script' not in schema"):
        apply_schema_overrides(tool, {"parameters": {"script": {"name": "renamed"}}})
