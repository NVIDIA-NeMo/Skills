# Tool Calling

Tool calling enables LLMs to execute external functions and use their results in generation. NeMo-Skills provides a flexible framework for both using built-in tools and creating custom ones.

## Overview

The tool calling system in NeMo-Skills is built on the [Model Context Protocol (MCP)](https://modelcontextprotocol.io/), which provides a standardized way to:

- Define tool schemas that LLMs can understand
- Execute tools with type-safe arguments
- Handle tool responses and integrate them back into the conversation

### Architecture

```
┌─────────────────┐
│      LLM        │  Generates tool calls based on available tools
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ToolManager    │  Routes calls to registered tools
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   MCPClientTool │  Communicates with MCP server
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   MCP Server    │  Executes actual tool logic
└─────────────────┘
```

## Using Built-in Tools

NeMo-Skills comes with several pre-built tools that you can use immediately.

### PythonTool

Executes Python code in a stateful Jupyter notebook environment.

**Command line usage:**

```bash
ns generate \
  --cluster local \
  --input_file data.jsonl \
  --output_dir outputs \
  --model Qwen/Qwen3-8B \
  --server_type vllm \
  --server_gpus 1 \
  --server_args '--enable-auto-tool-choice --tool-call-parser hermes' \
  --with_sandbox true \
  ++tool_modules=[nemo_skills.mcp.servers.python_tool.PythonTool] \
  ++inference.tokens_to_generate=8192 \
  ++inference.temperature=0.6

```

**Python API usage:**

```python
from nemo_skills.pipeline.cli import generate, wrap_arguments

generate(
    ctx=wrap_arguments(
        "++tool_modules=[nemo_skills.mcp.servers.python_tool.PythonTool] "
        "++inference.tokens_to_generate=8192 "
        "++inference.temperature=0.6"
    ),
    cluster='local',
    model='Qwen/Qwen3-8B',
    server_type='vllm',
    server_gpus=1,
    server_args='--enable-auto-tool-choice --tool-call-parser hermes',
    input_file='data.jsonl',
    output_dir='outputs',
    with_sandbox=True,  # Required for PythonTool
)
```

### Multiple Tools

You can use multiple tools simultaneously:

```bash
++tool_modules=[nemo_skills.mcp.servers.python_tool.PythonTool,nemo_skills.mcp.servers.exa_tool.ExaTool]
```

## Creating Custom Tools

Custom tools consist of two components:

1. **MCP Server** - Implements the actual tool logic
2. **Tool Class** - Client that connects to the server and can be configured via `tool_overrides`

### Example: Calculator Tool

Let's create a simple calculator tool that performs basic arithmetic operations.

#### Step 1: Create the MCP Server

Create `calculator_server.py`:

```python
"""MCP server that implements calculator functionality using sandbox execution."""
import argparse
from dataclasses import dataclass
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from omegaconf import OmegaConf
from pydantic import Field

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.mcp.utils import add_config_args, load_mcp_config

mcp = FastMCP(name="calculator_tool")

# Initialized from config in main()
sandbox = None


@dataclass
class CalculationResult:
    result: str = ""
    error: str | None = None


@mcp.tool(description="Perform mathematical calculations using Python")
async def calculate(
    operation: Annotated[
        str,
        Field(description="Operation to perform: add, subtract, multiply, or divide")
    ],
    x: Annotated[float, Field(description="First number")],
    y: Annotated[float, Field(description="Second number")],
    precision: Annotated[int, Field(description="Decimal precision")] = 2,
) -> CalculationResult:
    """Execute calculation in isolated sandbox environment."""

    # Map operation to Python operator
    op_symbols = {
        'add': '+',
        'subtract': '-',
        'multiply': '*',
        'divide': '/',
    }

    if operation not in op_symbols:
        return CalculationResult(error=f"Unknown operation: {operation}")

    # Generate Python code to execute in sandbox
    code = f"""
result = {x} {op_symbols[operation]} {y}
result = round(result, {precision})
print(f"{x} {operation} {y} = {{result}}")
"""

    try:
        # Execute in sandbox
        output_dict, session_id = await sandbox.execute_code(
            code,
            language="python",
            timeout=5.0,
        )

        if output_dict["process_status"] == "success":
            output = output_dict["stdout"].strip()
            return CalculationResult(result=output)
        else:
            error_msg = output_dict.get("stderr", "Execution failed")
            return CalculationResult(error=error_msg)

    except Exception as e:
        return CalculationResult(error=f"Execution error: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description="Calculator MCP server")
    add_config_args(parser)
    args = parser.parse_args()

    # Load sandbox configuration
    try:
        cfg = load_mcp_config(
            config=args.config,
            config_dir=args.config_dir,
            config_name=args.config_name,
        )
    except ValueError as e:
        # Fall back to default local sandbox
        cfg = OmegaConf.create({"sandbox": {"sandbox_type": "local"}})

    global sandbox
    sandbox_cfg = OmegaConf.to_container(cfg.sandbox, resolve=True)
    sandbox = get_sandbox(**sandbox_cfg)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
```

!!!note
    This example uses the NeMo-Skills sandbox for isolated code execution, similar to `PythonTool`. The sandbox provides security and isolation, making it suitable for executing untrusted or dynamic code.

#### Step 2: Create the Tool Class

Create `calculator_tool.py`:

```python
"""Calculator tool client for NeMo-Skills."""
from typing import Any, Dict

from nemo_skills.mcp.tool_providers import MCPClientTool


class CalculatorTool(MCPClientTool):
    """Tool for performing mathematical calculations."""

    def __init__(self) -> None:
        super().__init__()
        # Configure the MCP client to launch our server
        self.apply_config_updates(
            {
                "client": "nemo_skills.mcp.clients.MCPStdioClient",
                "client_params": {
                    "command": "python",
                    "args": ["/absolute/path/to/calculator_server.py"],
                },
                # Default precision that can be overridden
                "default_precision": 2,
            }
        )

    async def execute(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        extra_args: Dict[str, Any] | None = None
    ):
        """Execute the tool, injecting default precision if not provided."""
        arguments = dict(arguments)
        extra = dict(extra_args or {})

        if tool_name == "calculate":
            # Inject default precision via extra_args if not in arguments
            if "precision" not in arguments:
                extra["precision"] = self._config.get("default_precision", 2)

        return await self._client.call_tool(
            tool=tool_name,
            args=arguments,
            extra_args=extra
        )
```

#### Step 3: Use Your Custom Tool

**Command line:**

```bash
ns generate \
  --cluster local \
  --input_file data.jsonl \
  --output_dir outputs \
  --model Qwen/Qwen3-8B \
  --server_type vllm \
  --server_gpus 1 \
  --server_args '--enable-auto-tool-choice --tool-call-parser hermes' \
  ++tool_modules=[/absolute/path/to/calculator_tool.py::CalculatorTool] \
  ++tool_overrides.CalculatorTool.default_precision=4
```

**Python API:**

```python
from nemo_skills.pipeline.cli import generate, wrap_arguments

generate(
    ctx=wrap_arguments(
        "++tool_modules=[/absolute/path/to/calculator_tool.py::CalculatorTool] "
        "++tool_overrides.CalculatorTool.default_precision=4"
    ),
    cluster='local',
    model='Qwen/Qwen3-8B',
    server_type='vllm',
    server_gpus=1,
    server_args='--enable-auto-tool-choice --tool-call-parser hermes',
    input_file='data.jsonl',
    output_dir='outputs',
)
```

## Tool Configuration

### Tool Overrides

Tool overrides allow you to customize tool behavior without modifying code:

```bash
# Single override
++tool_overrides.CalculatorTool.default_precision=4

# Multiple overrides
++tool_overrides.CalculatorTool.default_precision=4 \
++tool_overrides.PythonTool.exec_timeout_s=30
```

### Hiding Arguments

You can hide arguments from the LLM's view while still passing them to the server:

```python
self.apply_config_updates({
    "hide_args": {
        "calculate": ["precision"]  # Hide precision from LLM schema
    },
})
```

The hidden argument is then injected via `extra_args` in the `execute()` method.

## Advanced Examples

### Using Multiple Tools Together

```python
from nemo_skills.pipeline.cli import generate, wrap_arguments

generate(
    ctx=wrap_arguments(
        "++tool_modules=["
        "nemo_skills.mcp.servers.python_tool.PythonTool,"
        "/path/to/calculator_tool.py::CalculatorTool,"
        "nemo_skills.mcp.servers.exa_tool.ExaTool"
        "] "
        "++tool_overrides.PythonTool.exec_timeout_s=30 "
        "++tool_overrides.CalculatorTool.default_precision=4"
    ),
    cluster='local',
    model='Qwen/Qwen3-8B',
    server_type='vllm',
    server_gpus=1,
    server_args='--enable-auto-tool-choice --tool-call-parser hermes',
    input_file='data.jsonl',
    output_dir='outputs',
    with_sandbox=True,
)
```

## Server Configuration

### vLLM Tool Calling

For vLLM, you may need to specify tool calling arguments:

```bash
--server_type vllm \
--server_args '--enable-auto-tool-choice --tool-call-parser hermes'
```


## Reference

### Built-in Tools

- [`nemo_skills.mcp.servers.python_tool.PythonTool`](https://github.com/NVIDIA-NeMo/Skills/tree/main/nemo_skills/mcp/servers/python_tool.py) - Python code execution
- [`nemo_skills.mcp.servers.exa_tool.ExaTool`](https://github.com/NVIDIA-NeMo/Skills/tree/main/nemo_skills/mcp/servers/exa_tool.py) - Web search via Exa API
