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
import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Annotated, Any
from urllib.parse import urlparse

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
TAVILY_API_KEYS: list[str] = []
TAVILY_REQUEST_COUNT: int = 0

MAX_NUM_RESULTS: int = 20
DEFAULT_NUM_RESULTS: int = 10
DEFAULT_SCROLL_WORDS: int = 2000
MAX_QUERY_CHARS: int = 400
MAX_RESULT_CHARS: int = 2000
PAGE_CACHE: dict[str, str] = {}

STATUS_CODE_ERRORS = {
    429: "Search rate limit exceeded",
    500: "Search request failed due to server error",
    502: "Search request failed due to bad gateway",
    503: "Search request failed due to service unavailable",
    504: "Search request failed due to gateway timeout",
}

# These errors should stop the process - no point continuing with bad credentials
FATAL_STATUS_CODES = {401, 403}
RETRY_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


## See docs https://docs.tavily.com/documentation/api-reference/endpoint/search
## There is also a hosted MCP that can be used instead of this tool: https://github.com/tavily-ai/tavily-mcp?tab=readme-ov-file#remote-mcp-server
@mcp.tool(name="web-search")
async def answer(
    query: Annotated[str, Field(description="Search query.")],
    exclude_domains: Annotated[list[str], Field(description="Domains to exclude from the search.")] = [],
    num_results: Annotated[int, Field(description="Number of results to return.")] = DEFAULT_NUM_RESULTS,
    answer_type: Annotated[
        str,
        Field(
            description=(
                'Type of results to return. Choose "formatted" for Gym-style text, '
                '"answer" for a concise answer, or "results" for raw search results.'
            )
        ),
    ] = "formatted",
):
    """Search the web for a query"""

    # Validate inputs
    if query is None:
        return "Query is none"
    if len(query) > MAX_QUERY_CHARS:
        return "Query is too long"
    if answer_type not in ["answer", "results", "formatted"]:
        return {"error": "Invalid answer type. Choose 'answer', 'results', or 'formatted'."}
    if num_results < 1 or num_results > MAX_NUM_RESULTS:
        return {"error": f"Number of results must be between 1 and {MAX_NUM_RESULTS}."}

    data = await _tavily_post(
        "/search",
        {
            "query": query,
            "search_depth": "advanced",
            "max_results": num_results,
            "exclude_domains": exclude_domains,
        },
    )
    if isinstance(data, dict) and data.get("error"):
        return data
    if answer_type == "formatted":
        return _postprocess_search_results(data)
    result = data.get(answer_type)
    if result is None:
        return {"error": "Search response is missing required field"}
    return result


@mcp.tool(name="find-in-page")
async def find_in_page(
    url: Annotated[str, Field(description="URL to fetch.")],
    query: Annotated[str, Field(description="Term or phrase to find on the page.")],
    exclude_domains: Annotated[list[str], Field(description="Domains to exclude from page retrieval.")] = [],
):
    """Fetch a page and return cleaned, line-numbered content relevant to a query."""

    if url is None:
        return "URL is none"
    if query is None:
        return "Query is none"
    if _is_url_excluded(url, exclude_domains):
        return "URL is in excluded domains"

    data = await _tavily_post("/extract", {"urls": url, "query": query})
    if isinstance(data, dict) and data.get("error"):
        return data

    raw_content = _extract_first_raw_content(data)
    if not raw_content:
        return "No content found."

    domain = _extract_domain(url)
    cleaned = _clean_text(raw_content)
    truncated, was_truncated = _truncate_text(cleaned)
    numbered = _add_line_numbers(truncated)

    header = f'Content from: {domain}\nURL: {url}\nQuery: "{query}"\n========================================\n'
    footer = "\n[...truncated, use scroll-page for full content]" if was_truncated else ""
    return header + numbered + footer


@mcp.tool(name="scroll-page")
async def scroll_page(
    url: Annotated[str, Field(description="URL to fetch.")],
    start_index: Annotated[int, Field(description="Zero-based word offset to start reading from.")] = 0,
    n: Annotated[int, Field(description="Number of words to return.")] = DEFAULT_SCROLL_WORDS,
    exclude_domains: Annotated[list[str], Field(description="Domains to exclude from page retrieval.")] = [],
):
    """Fetch a page and return a word-window of cleaned, line-numbered content."""

    if url is None:
        return {"results_string": "URL is none", "total_words": 0}
    if _is_url_excluded(url, exclude_domains):
        return {"results_string": "URL is in excluded domains", "total_words": 0}

    start_index = max(0, int(start_index or 0))
    n = max(1, int(n or DEFAULT_SCROLL_WORDS))

    if url in PAGE_CACHE:
        page_content = PAGE_CACHE[url]
    else:
        data = await _tavily_post("/extract", {"urls": url})
        if isinstance(data, dict) and data.get("error"):
            return data
        page_content = _extract_first_raw_content(data)
        PAGE_CACHE[url] = page_content

    words = page_content.split()
    total_words = len(words)
    end_index = min(start_index + n, total_words)
    chunk_text = " ".join(words[start_index:end_index])

    domain = _extract_domain(url)
    cleaned = _clean_text(chunk_text)
    numbered = _add_line_numbers(cleaned)
    header = (
        f"Page content from: {domain}\n"
        f"URL: {url}\n"
        f"Showing words [{start_index}-{end_index}] of {total_words}\n"
        f"========================================\n"
    )
    return {"results_string": header + numbered, "total_words": total_words}


async def _tavily_post(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    api_key = _select_tavily_api_key()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    url = f"https://api.tavily.com{endpoint}"

    max_num_tries = 3
    tries = 0
    last_response: httpx.Response | None = None
    while tries < max_num_tries:
        tries += 1
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException:
            return {"error": "Search request timed out"}
        except httpx.RequestError:
            return {"error": "Search request failed due to network error"}

        last_response = response
        if response.status_code in FATAL_STATUS_CODES:
            return {"error": "Search authentication failed", "fatal": True}
        if response.status_code in RETRY_STATUS_CODES:
            if response.status_code == 429:
                max_num_tries += 1
            await asyncio.sleep(0.5)
            continue
        if response.status_code != 200:
            error_msg = STATUS_CODE_ERRORS.get(
                response.status_code, f"Search request failed with status {response.status_code}"
            )
            return {"error": error_msg}

        try:
            return response.json()
        except json.JSONDecodeError:
            return {"error": "Search returned invalid response"}

    if last_response is not None:
        error_msg = STATUS_CODE_ERRORS.get(
            last_response.status_code, f"Search request failed with status {last_response.status_code}"
        )
        return {"error": error_msg}
    return {"error": "Search request failed"}


def _select_tavily_api_key() -> str:
    global TAVILY_REQUEST_COUNT
    if not TAVILY_API_KEYS:
        raise ValueError("Missing Tavily API key.")
    api_key = TAVILY_API_KEYS[TAVILY_REQUEST_COUNT % len(TAVILY_API_KEYS)]
    TAVILY_REQUEST_COUNT += 1
    return api_key


def _extract_first_raw_content(data: dict[str, Any]) -> str:
    results = data.get("results") or []
    if not results:
        return ""
    return results[0].get("raw_content") or ""


def _postprocess_search_results(results: dict[str, Any]) -> str:
    answer_text = results.get("answer")
    if answer_text is not None:
        return f"Search Answer\n==============\n{answer_text}\n"

    formatted_results = ["Search Results\n==============\n"]
    for i, result in enumerate(results.get("results") or [], 1):
        url = result.get("url", "")
        domain = _extract_domain(url)
        snippet = _clean_text(result.get("content", ""))
        snippet, _ = _truncate_text(snippet)
        formatted_results.append(
            f"[{i}] {result.get('title', '')} ({domain})\n    URL: {url}\n    Summary: {snippet}\n\n"
        )
    return "".join(formatted_results)


def _is_url_excluded(url: str, exclude_domains: list[str]) -> bool:
    hostname = urlparse(url).hostname or ""
    return any(hostname == domain or hostname.endswith("." + domain) for domain in exclude_domains)


def _extract_domain(url: str) -> str:
    return urlparse(url).hostname or url


def _clean_text(text: str) -> str:
    text = re.sub(r"\[edit\]", "", text)
    text = re.sub(r"^\[(?:Jump to content|Search|Read|Edit|View history)[^\]]*\].*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[[^\]]+\]\(https?://[a-z]{2,3}\.wikipedia\.org/[^\)]*\)", "", text)
    text = re.sub(r"^\s*\*\s*\[[^\]]*\]\(#[^\)]*\)\s*$", "", text, flags=re.MULTILINE)
    text = text.replace("\u200b", "").replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    text = text.replace("\u3010", "[").replace("\u3011", "]")
    text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _add_line_numbers(text: str) -> str:
    lines = text.split("\n")
    return "\n".join(f"L{i}: {line}" for i, line in enumerate(lines))


def _truncate_text(text: str, max_chars: int = MAX_RESULT_CHARS) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    cut = text.rfind("\n", 0, max_chars)
    if cut == -1:
        cut = max_chars
    return text[:cut], True


def _parse_api_keys(api_key_config: str | None) -> list[str]:
    if not api_key_config:
        return []
    return [key.strip() for key in api_key_config.split(",") if key.strip()]


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
                    "find-in-page": ["exclude_domains"],
                    "scroll-page": ["exclude_domains"],
                },
                "exclude_domains_config": None,
                "num_results": DEFAULT_NUM_RESULTS,
                "answer_type": "formatted",
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
        if tool_name == "web-search":
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

    global TAVILY_API_KEYS
    TAVILY_API_KEYS = _parse_api_keys(args.api_key)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
