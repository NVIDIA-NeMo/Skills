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
import os
from dataclasses import dataclass
from typing import Annotated

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from nemo_skills.mcp.tool_providers import MCPClientTool

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    error: str | None = None
    result: str | None = None


mcp = FastMCP(name="tavily")

# Populated from CLI args in main()
TAVILY_API_KEY: str | None = None


## See docs https://docs.tavily.com/documentation/api-reference/endpoint/search
## There is also a hosted MCP that can be used instead of this tool: https://github.com/tavily-ai/tavily-mcp?tab=readme-ov-file#remote-mcp-server
@mcp.tool(name="tavily-search")
async def answer(
    query: Annotated[str, Field(description="Search query.")],
):
    """Get a summary of search results from the web using Tavily."""

    api_url = "https://api.tavily.com/search"

    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "query": query,
        # "auto_parameters": False,
        "search_depth": "basic",
        "include_answer": "basic",  ## or advanced.
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(api_url, headers=headers, json=payload)
        if response.status_code != 200:
            return {"error": response.json()["error"]}

    result = response.json()["answer"]

    return result


class TavilySearchTool(MCPClientTool):
    def __init__(self) -> None:
        super().__init__()
        self.apply_config_updates(
            {
                "client": "nemo_skills.mcp.clients.MCPStdioClient",
                "client_params": {
                    "command": "python",
                    "args": ["-m", "nemo_skills.mcp.servers.tavily_search_tool"],
                },
            }
        )


def main():
    parser = argparse.ArgumentParser(description="MCP server for Tavily web search tool")
    parser.add_argument("--api-key", type=str, default=os.getenv("TAVILY_API_KEY"), help="Tavily API Key")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Missing Tavily API key.")

    global TAVILY_API_KEY
    TAVILY_API_KEY = args.api_key

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
