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
import json
import logging
import os
from dataclasses import dataclass
from typing import Annotated, Any

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

EXCLUDE_DOMAINS: list[str] | None = None


## See docs https://docs.tavily.com/documentation/api-reference/endpoint/search
## There is also a hosted MCP that can be used instead of this tool: https://github.com/tavily-ai/tavily-mcp?tab=readme-ov-file#remote-mcp-server
@mcp.tool(name="tavily-search")
async def answer(
    query: Annotated[str, Field(description="Search query.")],
    exclude_domains: Annotated[list[str], Field(description="Domains to exclude from the search.")] = [],
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
        # this should be statically set to the domains we want to exclude
        "exclude_domains": exclude_domains,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(api_url, headers=headers, json=payload)
        if response.status_code != 200:
            return {"error": response.json()["error"]}

    result = response.json()["answer"]

    return result


def _parse_exclude_domains(exclude_config: dict) -> list[str]:
    exclude_domains = []
    # this is pretty hard-coded so we ensure the file structure is correct
    notices = exclude_config["notices"]
    for notice in notices:
        for prop in notice["properties"]:
            if prop.get("type") == "domain":
                exclude_domains.append(prop["value"])
    return exclude_domains


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
                "hide_args": {
                    "tavily-search": ["exclude_domains"],
                },
                "exclude_domains_config": None,
            }
        )

    def post_configure(self) -> None:
        # Required the exclude domains to be set--we do not want to accidentally include all domains
        if (conf := self._config.get("exclude_domains_config")) is not None:
            with open(conf, "r") as f:
                exlude_config = json.load(f)
                self.exclude_domains = _parse_exclude_domains(exlude_config)
        else:
            raise ValueError("exclude_domains_config is not set")

    async def execute(self, tool_name: str, arguments: dict[str, Any], extra_args: dict[str, Any] | None = None):
        arguments = dict(arguments)
        merged_extra = dict(extra_args or {})
        if not hasattr(self, "exclude_domains"):
            raise ValueError("exclude_domains_config is not set")
        merged_extra["exclude_domains"] = self.exclude_domains
        result = await self._client.call_tool(tool=tool_name, args=arguments, extra_args=merged_extra)
        return result


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
