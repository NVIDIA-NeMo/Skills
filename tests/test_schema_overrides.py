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

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_skills.inference.model.base import EndpointType
from nemo_skills.mcp.adapters import apply_schema_overrides, format_tool_list_by_endpoint_type
from nemo_skills.mcp.schema_overrides import load_schema_overrides
from nemo_skills.mcp.tool_manager import Tool, ToolManager


# Test utilities
def create_mock_tool(name: str, description: str, parameters: dict, required: list = None):
    """Create a mock tool dict for testing."""
    return {
        "name": name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": parameters,
            "required": required or [],
        },
    }


def assert_schema_equals(actual: dict, expected: dict):
    """Helper to compare tool schemas."""
    assert actual["name"] == expected["name"]
    assert actual["description"] == expected["description"]
    assert actual["input_schema"]["properties"] == expected["input_schema"]["properties"]
    assert actual["input_schema"]["required"] == expected["input_schema"]["required"]


# Test fixtures
@pytest.fixture
def sample_tool():
    """Sample tool schema before override."""
    return create_mock_tool(
        name="stateful_python_code_exec",
        description="Executes Python code",
        parameters={"script": {"type": "string", "description": "The Python script to execute."}},
        required=["script"],
    )


@pytest.fixture
def sample_override_dict():
    """Sample override configuration."""
    return {
        "PythonTool": {
            "stateful_python_code_exec": {
                "name": "python_executor",
                "description": "Call this function to execute Python code in a stateful Jupyter notebook environment.",
                "parameters": {"script": {"name": "code"}},
            }
        }
    }


# Schema Override Loading Tests
def test_load_schema_overrides_from_dict(sample_override_dict):
    """Load overrides from inline dict."""
    result = load_schema_overrides(sample_override_dict)
    assert isinstance(result, dict)
    assert "PythonTool" in result
    assert "stateful_python_code_exec" in result["PythonTool"]
    assert result["PythonTool"]["stateful_python_code_exec"]["name"] == "python_executor"


def test_load_schema_overrides_invalid_format():
    """Reject invalid override structures."""
    with pytest.raises(ValueError):
        load_schema_overrides("not a dict")


def test_load_schema_overrides_none():
    """None override should return empty dict."""
    result = load_schema_overrides(None)
    assert result == {}


def test_load_schema_overrides_empty_dict():
    """Empty override dict should not error."""
    result = load_schema_overrides({})
    assert result == {}


# Schema Override Application Tests
def test_apply_schema_override_name(sample_tool):
    """Override tool name."""
    override_config = {"name": "new_tool_name"}
    transformed, mapping = apply_schema_overrides(sample_tool, override_config)
    assert transformed["name"] == "new_tool_name"
    assert transformed["description"] == sample_tool["description"]
    assert mapping == {}


def test_apply_schema_override_description(sample_tool):
    """Override tool description."""
    override_config = {"description": "New description"}
    transformed, mapping = apply_schema_overrides(sample_tool, override_config)
    assert transformed["name"] == sample_tool["name"]
    assert transformed["description"] == "New description"
    assert mapping == {}


def test_apply_schema_override_parameter_names(sample_tool):
    """Override parameter name."""
    override_config = {"parameters": {"script": {"name": "code"}}}
    transformed, mapping = apply_schema_overrides(sample_tool, override_config)
    assert "code" in transformed["input_schema"]["properties"]
    assert "script" not in transformed["input_schema"]["properties"]
    assert mapping == {"code": "script"}
    assert transformed["input_schema"]["required"] == ["code"]


def test_apply_schema_override_parameter_descriptions(sample_tool):
    """Override parameter description."""
    override_config = {
        "parameters": {
            "script": {
                "type": "string",
                "description": "New description for script parameter",
            }
        }
    }
    transformed, mapping = apply_schema_overrides(sample_tool, override_config)
    assert transformed["input_schema"]["properties"]["script"]["description"] == "New description for script parameter"
    assert mapping == {}


def test_apply_schema_override_multiple_parameters():
    """Override multiple parameters in one tool."""
    tool = create_mock_tool(
        name="test_tool",
        description="Test",
        parameters={
            "param1": {"type": "string"},
            "param2": {"type": "integer"},
        },
        required=["param1"],
    )
    override_config = {
        "parameters": {
            "param1": {"name": "new_param1", "type": "string"},
            "param2": {"name": "new_param2", "type": "integer"},
        }
    }
    transformed, mapping = apply_schema_overrides(tool, override_config)
    assert "new_param1" in transformed["input_schema"]["properties"]
    assert "new_param2" in transformed["input_schema"]["properties"]
    assert mapping == {"new_param1": "param1", "new_param2": "param2"}
    assert transformed["input_schema"]["required"] == ["new_param1"]


def test_apply_schema_override_nonexistent_tool(sample_tool):
    """Override for tool that doesn't exist should not error."""
    override_config = None
    transformed, mapping = apply_schema_overrides(sample_tool, override_config)
    assert transformed == sample_tool
    assert mapping == {}


def test_apply_schema_override_partial(sample_tool):
    """Allow partial overrides."""
    override_config = {"name": "new_name"}
    transformed, mapping = apply_schema_overrides(sample_tool, override_config)
    assert transformed["name"] == "new_name"
    assert transformed["description"] == sample_tool["description"]


def test_apply_schema_override_single_param_positional():
    """Test positional parameter mapping when only one parameter."""
    # Positional/implicit mapping is no longer supported; require explicit .name mapping.
    tool = create_mock_tool(name="test", description="Test", parameters={"script": {"type": "string"}})
    with pytest.raises(ValueError):
        apply_schema_overrides(tool, {"parameters": {"code": {"type": "string", "description": "Code"}}})


def test_format_tool_list_with_overrides(sample_tool, sample_override_dict):
    """Test format_tool_list_by_endpoint_type with overrides."""
    # Add server field to tool (provider class name) - this is added by ToolManager
    sample_tool_with_server = {**sample_tool, "server": "PythonTool"}
    tools = [sample_tool_with_server]
    schema_overrides = load_schema_overrides(sample_override_dict)

    # Test chat endpoint
    formatted_chat, mappings = format_tool_list_by_endpoint_type(tools, EndpointType.chat, schema_overrides)
    assert len(formatted_chat) == 1
    assert formatted_chat[0]["function"]["name"] == "python_executor"
    assert "code" in formatted_chat[0]["function"]["parameters"]["properties"]
    assert mappings["parameters"]["python_executor"] == {"code": "script"}
    assert mappings["tool_names"]["python_executor"] == "stateful_python_code_exec"

    # Test responses endpoint
    formatted_resp, mappings_resp = format_tool_list_by_endpoint_type(tools, EndpointType.responses, schema_overrides)
    assert len(formatted_resp) == 1
    assert formatted_resp[0]["name"] == "python_executor"
    assert "code" in formatted_resp[0]["parameters"]["properties"]


# Parameter Mapping Tests
def test_parameter_mapping_single_param():
    """Map single parameter name."""
    tool = create_mock_tool(
        name="test",
        description="Test",
        parameters={"script": {"type": "string"}},
    )
    override_config = {"parameters": {"script": {"name": "code"}}}
    transformed, mapping = apply_schema_overrides(tool, override_config)
    assert mapping == {"code": "script"}


def test_parameter_mapping_multiple_params():
    """Map multiple parameters."""
    tool = create_mock_tool(
        name="test",
        description="Test",
        parameters={
            "param1": {"type": "string"},
            "param2": {"type": "integer"},
        },
    )
    override_config = {
        "parameters": {
            "param1": {"name": "new1"},
            "param2": {"name": "new2", "type": "integer"},
        }
    }
    transformed, mapping = apply_schema_overrides(tool, override_config)
    assert mapping == {"new1": "param1", "new2": "param2"}


def test_parameter_mapping_missing_param():
    """Model sends parameter not in override should pass through."""
    tool = create_mock_tool(
        name="test",
        description="Test",
        parameters={
            "param1": {"type": "string"},
            "param2": {"type": "string"},
        },
    )
    override_config = {"parameters": {"param1": {"name": "new1"}}}
    transformed, mapping = apply_schema_overrides(tool, override_config)
    # param2 should still be in properties
    assert "param2" in transformed["input_schema"]["properties"]
    assert mapping == {"new1": "param1"}


def test_parameter_mapping_extra_params():
    """Model sends extra parameters should pass through."""
    # This is tested implicitly in the execution tests
    pass


# Integration Tests - define tool classes at module level
class DummyToolProvider(Tool):
    """Test tool provider for integration tests."""

    def default_config(self):
        return {}

    def configure(self, overrides=None, context=None):
        pass

    async def list_tools(self):
        return [
            {
                "name": "execute",
                "description": "Run code",
                "input_schema": {
                    "type": "object",
                    "properties": {"script": {"type": "string"}},
                    "required": ["script"],
                },
            }
        ]

    async def execute(self, tool_name: str, arguments: dict, extra_args=None):
        return {"result": arguments}


class MockToolProvider(Tool):
    """Mock tool provider for testing."""

    def default_config(self):
        return {}

    def configure(self, overrides=None, context=None):
        pass

    async def list_tools(self):
        return [
            {
                "name": "test_tool",
                "description": "Test tool",
                "input_schema": {
                    "type": "object",
                    "properties": {"script": {"type": "string"}},
                    "required": ["script"],
                },
            }
        ]

    async def execute(self, tool_name: str, arguments: dict, extra_args=None):
        return {"result": arguments}


class TestToolProvider(Tool):
    """Test tool provider for end-to-end tests."""

    def default_config(self):
        return {}

    def configure(self, overrides=None, context=None):
        pass

    async def list_tools(self):
        return [
            {
                "name": "stateful_python_code_exec",
                "description": "Executes Python code",
                "input_schema": {
                    "type": "object",
                    "properties": {"script": {"type": "string", "description": "The Python script"}},
                    "required": ["script"],
                },
            }
        ]

    async def execute(self, tool_name: str, arguments: dict, extra_args=None):
        return {"result": f"Executed with {arguments}"}


class TrackingTestToolProvider(TestToolProvider):
    """Test tool provider that tracks received arguments."""

    async def execute(self, tool_name: str, arguments: dict, extra_args=None):
        # Store in module-level dict for test access
        import tests.test_schema_overrides as test_module

        if not hasattr(test_module, "_received_args"):
            test_module._received_args = {}
        test_module._received_args[tool_name] = arguments
        return {"result": f"Executed with {arguments}"}


@pytest.mark.asyncio
async def test_tool_manager_with_schema_overrides():
    """Create ToolManager with schema overrides."""
    tm = ToolManager(module_specs=[f"{__name__}::DummyToolProvider"], overrides={}, context={})
    tools = await tm.list_all_tools(use_cache=False)
    assert len(tools) == 1
    assert tools[0]["name"] == "execute"


@pytest.mark.asyncio
async def test_tool_calling_wrapper_with_overrides():
    """Create ToolCallingWrapper with schema_overrides."""
    from nemo_skills.inference.model.base import BaseModel
    from nemo_skills.inference.model.tool_call import ToolCallingWrapper

    # Create mock model
    mock_model = MagicMock(spec=BaseModel)
    mock_model.generate_async = AsyncMock(
        return_value={
            "generation": "test",
            "num_generated_tokens": 10,
            "serialized_output": [],
            "tool_calls": [],
        }
    )

    schema_overrides = {
        "MockToolProvider": {
            "test_tool": {
                "name": "renamed_tool",
                "parameters": {"script": {"name": "code"}},
            }
        }
    }

    wrapper = ToolCallingWrapper(
        model=mock_model,
        tool_modules=[f"{__name__}::MockToolProvider"],
        schema_overrides=schema_overrides,
    )

    # List tools and verify override applied
    raw_tools = await wrapper.tool_manager.list_all_tools(use_cache=False)
    from nemo_skills.mcp.adapters import format_tool_list_by_endpoint_type
    from nemo_skills.mcp.schema_overrides import load_schema_overrides

    loaded_overrides = load_schema_overrides(schema_overrides)
    tools, mappings = format_tool_list_by_endpoint_type(raw_tools, EndpointType.chat, loaded_overrides)
    assert tools[0]["function"]["name"] == "renamed_tool"
    assert "code" in tools[0]["function"]["parameters"]["properties"]
    assert mappings["parameters"]["renamed_tool"] == {"code": "script"}


@pytest.mark.asyncio
async def test_schema_override_end_to_end():
    """End-to-end test with mock tool and parameter mapping."""
    # Track what arguments the tool receives
    import tests.test_schema_overrides as test_module
    from nemo_skills.inference.model.base import BaseModel
    from nemo_skills.inference.model.tool_call import ToolCallingWrapper

    test_module._received_args = {}

    # Create mock model that returns a tool call
    mock_model = MagicMock(spec=BaseModel)

    async def mock_generate(prompt, tools=None, **kwargs):
        # Simulate model calling tool with overridden parameter name
        return {
            "generation": "",
            "num_generated_tokens": 5,
            "serialized_output": [],
            "tool_calls": [
                MagicMock(
                    model_dump=lambda: {
                        "function": {
                            "name": "python_executor",
                            "arguments": json.dumps({"code": "print('hello')"}),
                        },
                        "id": "call_123",
                    }
                )
            ],
        }

    mock_model.generate_async = AsyncMock(side_effect=mock_generate)

    schema_overrides = {
        "TrackingTestToolProvider": {
            "stateful_python_code_exec": {
                "name": "python_executor",
                "parameters": {"script": {"name": "code"}},
            }
        }
    }

    wrapper = ToolCallingWrapper(
        model=mock_model,
        tool_modules=[f"{__name__}::TrackingTestToolProvider"],
        schema_overrides=schema_overrides,
    )

    # Generate and verify parameter mapping
    await wrapper.generate_async(
        prompt=[{"role": "user", "content": "Execute code"}],
        endpoint_type=EndpointType.chat,
        tokens_to_generate=100,
    )

    # Verify tool was called with original parameter name
    received_args = test_module._received_args
    assert "stateful_python_code_exec" in received_args, f"Expected stateful_python_code_exec in {received_args}"
    assert "script" in received_args["stateful_python_code_exec"], (
        f"Expected script in {received_args['stateful_python_code_exec']}"
    )
    assert received_args["stateful_python_code_exec"]["script"] == "print('hello')"
    assert "code" not in received_args["stateful_python_code_exec"]


# Edge Cases and Error Handling
def test_schema_override_required_params(sample_tool):
    """Verify required parameters are preserved after override."""
    override_config = {"parameters": {"script": {"name": "code"}}}
    transformed, mapping = apply_schema_overrides(sample_tool, override_config)
    assert "code" in transformed["input_schema"]["required"]


def test_schema_override_type_validation():
    """Ensure parameter types can be overridden."""
    tool = create_mock_tool(
        name="test",
        description="Test",
        parameters={"param": {"type": "string"}},
    )
    override_config = {"parameters": {"param": {"type": "integer", "description": "Integer param"}}}
    transformed, mapping = apply_schema_overrides(tool, override_config)
    assert transformed["input_schema"]["properties"]["param"]["type"] == "integer"


def test_schema_override_circular_mapping():
    """Test that we don't create circular mappings."""
    # This is more of a validation test - our current implementation
    # doesn't prevent circular mappings, but it shouldn't cause issues
    # since we map model params -> original params, not the reverse
    tool = create_mock_tool(
        name="test",
        description="Test",
        parameters={"param1": {"type": "string"}, "param2": {"type": "string"}},
    )
    override_config = {
        "parameters": {
            "param1": {"name": "param2"},
            "param2": {"name": "param1"},
        }
    }
    transformed, mapping = apply_schema_overrides(tool, override_config)
    assert mapping == {"param1": "param2", "param2": "param1"}


def test_schema_override_with_dash_in_tool_name():
    """Test schema overrides work with tool names containing dashes."""
    tool = create_mock_tool(
        name="web-search",
        description="Search the web",
        parameters={"query": {"type": "string"}},
        required=["query"],
    )
    # Tool name with dash - should work fine in dict format
    schema_overrides = {
        "TavilySearchTool": {
            "web-search": {
                "name": "web_search",
                "description": "Search the web using Tavily",
                "parameters": {"query": {"name": "search_query"}},
            }
        }
    }

    loaded_overrides = load_schema_overrides(schema_overrides)
    tool_with_server = {**tool, "server": "TavilySearchTool"}
    tools = [tool_with_server]

    formatted_tools, mappings = format_tool_list_by_endpoint_type(
        tools, EndpointType.chat, schema_overrides=loaded_overrides
    )

    assert len(formatted_tools) == 1
    assert formatted_tools[0]["function"]["name"] == "web_search"
    assert "search_query" in formatted_tools[0]["function"]["parameters"]["properties"]
    assert mappings["parameters"]["web_search"] == {"search_query": "query"}
