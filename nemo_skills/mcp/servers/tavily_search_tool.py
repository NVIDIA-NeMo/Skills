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

from nemo_skills.mcp.tool_manager import FatalToolError
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
MAX_NUM_RESULTS: int = 20

STATUS_CODE_ERRORS = {
    429: "Search rate limit exceeded",
    500: "Search request failed due to server error",
    502: "Search request failed due to bad gateway",
    503: "Search request failed due to service unavailable",
    504: "Search request failed due to gateway timeout",
}

# These errors should stop the process - no point continuing with bad credentials
FATAL_STATUS_CODES = {401, 403}


## See docs https://docs.tavily.com/documentation/api-reference/endpoint/search
## There is also a hosted MCP that can be used instead of this tool: https://github.com/tavily-ai/tavily-mcp?tab=readme-ov-file#remote-mcp-server
@mcp.tool(name="web-search")
async def answer(
    query: Annotated[str, Field(description="Search query.")],
    exclude_domains: Annotated[list[str], Field(description="Domains to exclude from the search.")] = [],
    num_results: Annotated[int, Field(description="Number of results to return.")] = 10,
    answer_type: Annotated[
        str,
        Field(
            description='Type of results to return. Choose "answer" for a concise answer or "results" for a list of results.'
        ),
    ] = "answer",
):
    """Search the web for a query"""

    api_url = "https://api.tavily.com/search"

    # Validate inputs
    if answer_type not in ["answer", "results"]:
        return {"error": "Invalid answer type. Choose 'answer' or 'results'."}
    if num_results > MAX_NUM_RESULTS:
        return {"error": f"Number of results must be less than or equal to {MAX_NUM_RESULTS}."}

    headers = {
        "Authorization": f"Bearer {TAVILY_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "query": query,
        # "auto_parameters": False,
        "search_depth": "basic",
        "include_answer": "basic",  ## or advanced.
        "num_results": num_results,
        # this should be statically set to the domains we want to exclude
        "exclude_domains": exclude_domains,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(api_url, headers=headers, json=payload)
    except httpx.TimeoutException:
        return {"error": "Search request timed out"}
    except httpx.RequestError:
        return {"error": "Search request failed due to network error"}

    # Handle non-200 responses
    if response.status_code in FATAL_STATUS_CODES:
        return {"error": "Search authentication failed", "fatal": True}
    if response.status_code != 200:
        error_msg = STATUS_CODE_ERRORS.get(
            response.status_code, f"Search request failed with status {response.status_code}"
        )
        return {"error": error_msg}

    # Parse response
    try:
        data = response.json()
    except json.JSONDecodeError:
        return {"error": "Search returned invalid response"}

    # Extract result
    result = data.get(answer_type)
    if result is None:
        return {"error": "Search response is missing required field"}

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
                    "web-search": ["exclude_domains", "num_results", "answer_type"],
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
        for key in ["num_results", "answer_type"]:
            if key in self._config:
                merged_extra[key] = self._config[key]
        result = await self._client.call_tool(tool=tool_name, args=arguments, extra_args=merged_extra)

        # Check for fatal errors that should stop the process
        if isinstance(result, dict) and result.get("fatal"):
            raise FatalToolError(result.get("error", "Fatal tool error"))

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
