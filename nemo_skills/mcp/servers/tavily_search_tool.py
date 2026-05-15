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

"""Tavily MCP and direct tools.

The direct tools mirror the ``PythonTool`` -> ``DirectPythonTool`` split:
they implement the Tool abstraction directly and call Tavily over HTTP without
spawning an MCP server/client pair.

Usage:
    ++tool_modules=[nemo_skills.mcp.servers.tavily_search_tool::DirectTavilySearchTool]
    ++tool_modules=[nemo_skills.mcp.servers.tavily_search_tool::DirectTavilyGymTool]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
from collections import defaultdict
from dataclasses import dataclass, field
from time import time
from typing import Annotated, Any
from urllib.parse import urlparse

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from nemo_skills.mcp.tool_manager import FatalToolError, Tool
from nemo_skills.mcp.tool_providers import MCPClientTool

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    error: str | None = None
    result: str | None = None


@dataclass
class TavilySearchSingleCallMetric:
    function: str
    status: str
    start_time: float
    end_time: float
    time_taken: float = field(init=False)

    def __post_init__(self) -> None:
        self.time_taken = self.end_time - self.start_time


@dataclass
class TavilySearchMetrics:
    async_tavily_calls: list[TavilySearchSingleCallMetric] = field(default_factory=list)


mcp = FastMCP(name="tavily")

# Populated from CLI args in main()
TAVILY_API_KEY: str | None = None

EXCLUDE_DOMAINS: list[str] | None = None
MAX_NUM_RESULTS: int = 20
DEFAULT_API_BASE_URL = "https://api.tavily.com"
DEFAULT_HTTP_TIMEOUT_S = 30.0
DEFAULT_MAX_RETRIES = 2
DEFAULT_RETRY_BACKOFF_S = 0.5
DEFAULT_GYM_MAX_RESULTS = 10
DEFAULT_GYM_MAX_RESULT_CHARS = 2000
DEFAULT_GYM_MAX_QUERY_CHARS = 400
DEFAULT_SCROLL_WORDS = 2000

STATUS_CODE_ERRORS = {
    400: "Search request is invalid",
    429: "Search rate limit exceeded",
    432: "Search request failed due to Tavily account quota or billing status",
    433: "Search request failed due to Tavily access restrictions",
    500: "Search request failed due to server error",
    502: "Search request failed due to bad gateway",
    503: "Search request failed due to service unavailable",
    504: "Search request failed due to gateway timeout",
}

RETRY_STATUS_CODES = {429, 500, 502, 503, 504}

# These errors should stop the process - no point continuing with bad credentials
FATAL_STATUS_CODES = {401, 403}


def _parse_exclude_domains(exclude_config: dict[str, Any]) -> list[str]:
    exclude_domains = []
    # This is deliberately aligned with the Nemotron/TDM registry structure.
    notices = exclude_config["notices"]
    for notice in notices:
        for prop in notice["properties"]:
            if prop.get("type") == "domain":
                exclude_domains.append(prop["value"])
    return exclude_domains


def _load_exclude_domains_from_file(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        exclude_config = json.load(f)
    return _parse_exclude_domains(exclude_config)


def _normalize_api_keys(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        value = os.getenv("TAVILY_API_KEY")
    if value is None:
        return []
    if isinstance(value, str):
        return [key.strip() for key in value.split(",") if key.strip()]
    return [str(key).strip() for key in value if str(key).strip()]


def _retry_after_seconds(response: httpx.Response) -> float | None:
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, min(float(raw), 30.0))
    except ValueError:
        return None


async def _post_tavily_json(
    *,
    api_base_url: str,
    api_key: str,
    endpoint: str,
    payload: dict[str, Any],
    timeout_s: float,
    max_retries: int,
    retry_backoff_s: float,
) -> dict[str, Any]:
    url = f"{api_base_url.rstrip('/')}/{endpoint.lstrip('/')}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    delay = retry_backoff_s
    last_error = "Search request failed"

    for attempt in range(max_retries + 1):
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                response = await client.post(url, headers=headers, json=payload)
        except httpx.TimeoutException:
            last_error = "Search request timed out"
            if attempt < max_retries:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            return {"error": last_error}
        except httpx.RequestError:
            last_error = "Search request failed due to network error"
            if attempt < max_retries:
                await asyncio.sleep(delay)
                delay *= 2
                continue
            return {"error": last_error}

        if response.status_code in FATAL_STATUS_CODES:
            raise FatalToolError("Search authentication failed")

        if response.status_code == 200:
            try:
                return response.json()
            except json.JSONDecodeError:
                return {"error": "Search returned invalid response"}

        last_error = STATUS_CODE_ERRORS.get(
            response.status_code,
            f"Search request failed with status {response.status_code}",
        )
        if response.status_code in RETRY_STATUS_CODES and attempt < max_retries:
            await asyncio.sleep(_retry_after_seconds(response) or delay)
            delay *= 2
            continue
        return {"error": last_error}

    return {"error": last_error}


def _extract_domain(url: str) -> str:
    return urlparse(url).hostname or url


def _is_url_excluded(url: str, exclude_domains: list[str]) -> bool:
    hostname = urlparse(url).hostname or ""
    return any(hostname == domain or hostname.endswith("." + domain) for domain in exclude_domains)


def _clean_text(text: str) -> str:
    """Remove common wiki/web navigation artifacts and normalize whitespace."""
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


def _truncate_text(text: str, max_chars: int = DEFAULT_GYM_MAX_RESULT_CHARS) -> tuple[str, bool]:
    if len(text) <= max_chars:
        return text, False
    cut = text.rfind("\n", 0, max_chars)
    if cut == -1:
        cut = max_chars
    return text[:cut], True


def _first_raw_content(results: dict[str, Any]) -> str:
    result_list = results.get("results") or []
    if not result_list:
        return ""
    return result_list[0].get("raw_content", "") or ""


def _postprocess_gym_search_results(results: dict[str, Any], max_result_chars: int) -> list[str]:
    answer = results.get("answer")
    if answer is not None:
        return [f"Search Answer\n==============\n{answer}\n"]

    formatted_results = ["Search Results\n==============\n"]
    for i, result in enumerate(results.get("results") or [], 1):
        domain = _extract_domain(result.get("url", ""))
        snippet = _clean_text(result.get("content", ""))
        snippet, _ = _truncate_text(snippet, max_result_chars)
        formatted_results.append(
            f"[{i}] {result.get('title', '')} ({domain})\n URL: {result.get('url', '')}\n Summary: {snippet}\n\n"
        )
    return formatted_results


class _DirectTavilyBase(Tool):
    def __init__(self) -> None:
        self._config: dict[str, Any] = {
            "tavily_api_key": None,
            "api_base_url": DEFAULT_API_BASE_URL,
            "exclude_domains": [],
            "exclude_domains_config": None,
            "require_exclude_domains_config": False,
            "timeout_s": DEFAULT_HTTP_TIMEOUT_S,
            "max_retries": DEFAULT_MAX_RETRIES,
            "retry_backoff_s": DEFAULT_RETRY_BACKOFF_S,
            "hide_args": {},
        }
        self._api_keys: list[str] = []
        self._num_requests = 0
        self._exclude_domains: list[str] = []
        self._sanitize_keys: dict[str, set[str]] = {}
        self.requests_to_metrics: dict[str, TavilySearchMetrics] = defaultdict(TavilySearchMetrics)

    def default_config(self) -> dict[str, Any]:
        return dict(self._config)

    def configure(self, overrides: dict[str, Any] | None = None, context: dict[str, Any] | None = None) -> None:
        cfg = dict(self._config)
        if overrides:
            unknown = set(overrides) - set(cfg)
            if unknown:
                raise ValueError(f"Unknown {self.__class__.__name__} override(s): {sorted(unknown)}")
            cfg.update(overrides)

        self._config = cfg
        self._api_keys = _normalize_api_keys(cfg.get("tavily_api_key"))
        self._exclude_domains = list(cfg.get("exclude_domains") or [])
        if cfg.get("exclude_domains_config"):
            self._exclude_domains.extend(_load_exclude_domains_from_file(cfg["exclude_domains_config"]))
        if cfg.get("require_exclude_domains_config") and not self._exclude_domains:
            raise ValueError("exclude_domains_config is not set")
        self._exclude_domains = sorted(set(self._exclude_domains))
        self._sanitize_keys = {tool: set(keys) for tool, keys in cfg.get("hide_args", {}).items()}

    async def list_tools(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def execute(
        self, tool_name: str, arguments: dict[str, Any], extra_args: dict[str, Any] | None = None
    ) -> Any:
        raise NotImplementedError

    def _select_api_key(self) -> str:
        if not self._api_keys:
            raise FatalToolError("Missing Tavily API key. Set TAVILY_API_KEY or tool override tavily_api_key.")
        api_key = self._api_keys[self._num_requests % len(self._api_keys)]
        self._num_requests += 1
        return api_key

    async def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        return await _post_tavily_json(
            api_base_url=self._config["api_base_url"],
            api_key=self._select_api_key(),
            endpoint=endpoint,
            payload=payload,
            timeout_s=float(self._config["timeout_s"]),
            max_retries=int(self._config["max_retries"]),
            retry_backoff_s=float(self._config["retry_backoff_s"]),
        )

    def _sanitize_arguments(self, tool_name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
        hidden = self._sanitize_keys.get(tool_name, set())
        return {k: v for k, v in dict(arguments or {}).items() if k not in hidden}

    async def _tracked_post_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
        *,
        request_id: str | None,
        function: str,
    ) -> dict[str, Any]:
        start_time = time()
        status = "success"
        try:
            response = await self._post_json(endpoint, payload)
            if isinstance(response, dict) and response.get("error"):
                status = "error"
            return response
        except Exception:
            status = "error"
            raise
        finally:
            if request_id is not None:
                self.requests_to_metrics[request_id].async_tavily_calls.append(
                    TavilySearchSingleCallMetric(
                        function=function,
                        status=status,
                        start_time=start_time,
                        end_time=time(),
                    )
                )

    async def cleanup_request(self, request_id: str) -> None:
        self.requests_to_metrics.pop(request_id, None)


## See docs https://docs.tavily.com/documentation/api-reference/endpoint/search
## There is also a hosted MCP that can be used instead of this tool:
## https://github.com/tavily-ai/tavily-mcp?tab=readme-ov-file#remote-mcp-server
@mcp.tool(name="web-search")
async def answer(
    query: Annotated[str, Field(description="Search query.")],
    exclude_domains: Annotated[list[str] | None, Field(description="Domains to exclude from the search.")] = None,
    num_results: Annotated[int, Field(description="Number of results to return.")] = 10,
    answer_type: Annotated[
        str,
        Field(
            description='Type of results to return. Choose "answer" for a concise answer or "results" for a list of results.'
        ),
    ] = "answer",
):
    """Search the web for a query."""

    # Validate inputs
    if answer_type not in ["answer", "results"]:
        return {"error": "Invalid answer type. Choose 'answer' or 'results'."}
    if num_results > MAX_NUM_RESULTS:
        return {"error": f"Number of results must be less than or equal to {MAX_NUM_RESULTS}."}

    payload = {
        "query": query,
        "search_depth": "basic",
        "include_answer": "basic",
        "max_results": num_results,
        "exclude_domains": exclude_domains or [],
    }

    try:
        result = await _post_tavily_json(
            api_base_url=DEFAULT_API_BASE_URL,
            api_key=TAVILY_API_KEY or "",
            endpoint="/search",
            payload=payload,
            timeout_s=DEFAULT_HTTP_TIMEOUT_S,
            max_retries=DEFAULT_MAX_RETRIES,
            retry_backoff_s=DEFAULT_RETRY_BACKOFF_S,
        )
    except FatalToolError:
        return {"error": "Search authentication failed", "fatal": True}
    if isinstance(result, dict) and result.get("error"):
        return result

    extracted = result.get(answer_type)
    if extracted is None:
        return {"error": "Search response is missing required field"}

    return extracted


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
        # Require excluded domains for the legacy MCP path to preserve existing behavior.
        if (conf := self._config.get("exclude_domains_config")) is not None:
            self.exclude_domains = _load_exclude_domains_from_file(conf)
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


class DirectTavilySearchTool(_DirectTavilyBase):
    """Direct version of TavilySearchTool that bypasses MCP transport."""

    def __init__(self) -> None:
        super().__init__()
        self._config.update(
            {
                "hide_args": {
                    "web-search": ["exclude_domains", "num_results", "answer_type"],
                    "web_search": ["exclude_domains", "num_results", "answer_type"],
                },
                "num_results": 10,
                "answer_type": "answer",
                "search_depth": "basic",
                "include_answer": "basic",
                "max_num_results": MAX_NUM_RESULTS,
            }
        )

    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "web_search",
                "description": "Search the web for a query using Tavily.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query."}},
                    "required": ["query"],
                },
            }
        ]

    async def execute(
        self, tool_name: str, arguments: dict[str, Any], extra_args: dict[str, Any] | None = None
    ) -> Any:
        if tool_name not in {"web-search", "web_search"}:
            return {"error": f"unknown tool '{tool_name}'"}

        arguments = self._sanitize_arguments(tool_name, arguments)
        query = arguments.get("query")
        if not isinstance(query, str):
            return {"error": "Missing required argument 'query'"}

        num_results = int(self._config["num_results"])
        if num_results > int(self._config["max_num_results"]):
            return {"error": f"Number of results must be less than or equal to {self._config['max_num_results']}."}

        answer_type = self._config["answer_type"]
        if answer_type not in ["answer", "results"]:
            return {"error": "Invalid answer type. Choose 'answer' or 'results'."}

        extra_args = dict(extra_args or {})
        request_id = extra_args.get("request_id")
        payload = {
            "query": query,
            "search_depth": self._config["search_depth"],
            "include_answer": self._config["include_answer"],
            "max_results": num_results,
            "exclude_domains": self._exclude_domains,
        }
        result = await self._tracked_post_json("/search", payload, request_id=request_id, function="search")
        if isinstance(result, dict) and result.get("error"):
            return result

        extracted = result.get(answer_type)
        if extracted is None:
            return {"error": "Search response is missing required field"}
        return extracted


class DirectTavilyGymTool(_DirectTavilyBase):
    """Direct implementation of the NeMo-Gym Tavily search resource tools.

    This ports the core tool surface from NVIDIA-NeMo/Gym's Tavily resource
    server: web search, query-focused page extraction, and paginated page
    scrolling. Judge/verify logic is intentionally excluded.
    """

    def __init__(self) -> None:
        super().__init__()
        self._config.update(
            {
                "hide_args": {
                    "web_search": ["exclude_domains", "max_results", "search_depth"],
                    "find_in_page": ["extract_depth"],
                    "scroll_page": ["extract_depth"],
                },
                "max_results": DEFAULT_GYM_MAX_RESULTS,
                "max_result_chars": DEFAULT_GYM_MAX_RESULT_CHARS,
                "max_query_chars": DEFAULT_GYM_MAX_QUERY_CHARS,
                "search_depth": "advanced",
                "include_answer": None,
                "extract_depth": "basic",
                "extract_format": "markdown",
                "scroll_words": DEFAULT_SCROLL_WORDS,
            }
        )
        self._page_cache: dict[str, str] = {}

    def configure(self, overrides: dict[str, Any] | None = None, context: dict[str, Any] | None = None) -> None:
        super().configure(overrides, context)
        self._page_cache = {}

    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "web_search",
                "description": "Search the web using Tavily and return a concise answer or ranked search results.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query."}},
                    "required": ["query"],
                },
            },
            {
                "name": "find_in_page",
                "description": "Find query-relevant content inside a specific URL using Tavily Extract.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to inspect."},
                        "query": {"type": "string", "description": "Term or question to locate within the page."},
                    },
                    "required": ["url", "query"],
                },
            },
            {
                "name": "scroll_page",
                "description": "Read a word window from a URL, starting at a specific word index.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "URL to read."},
                        "start_index": {
                            "type": "integer",
                            "description": "Zero-based word index to start reading from.",
                        },
                        "n": {"type": "integer", "description": "Number of words to return."},
                    },
                    "required": ["url"],
                },
            },
        ]

    async def execute(
        self, tool_name: str, arguments: dict[str, Any], extra_args: dict[str, Any] | None = None
    ) -> str | dict[str, Any]:
        arguments = self._sanitize_arguments(tool_name, arguments)
        request_id = (extra_args or {}).get("request_id")
        if tool_name == "web_search":
            return await self._web_search(arguments, request_id=request_id)
        if tool_name == "find_in_page":
            return await self._find_in_page(arguments, request_id=request_id)
        if tool_name == "scroll_page":
            return await self._scroll_page(arguments, request_id=request_id)
        return f"Error: unknown tool '{tool_name}'"

    async def _web_search(self, arguments: dict[str, Any], *, request_id: str | None) -> str:
        query = arguments.get("query")
        if query is None:
            return "Query is none"
        if not isinstance(query, str):
            return "Query must be a string"
        if len(query) > int(self._config["max_query_chars"]):
            return "Query is too long"

        payload = {
            "query": query,
            "max_results": int(self._config["max_results"]),
            "exclude_domains": self._exclude_domains,
            "search_depth": self._config["search_depth"],
        }
        if self._config["include_answer"] is not None:
            payload["include_answer"] = self._config["include_answer"]
        results = await self._tracked_post_json("/search", payload, request_id=request_id, function="search")
        if results.get("error"):
            return results["error"]
        return "".join(_postprocess_gym_search_results(results, int(self._config["max_result_chars"])))

    async def _find_in_page(self, arguments: dict[str, Any], *, request_id: str | None) -> str:
        url = arguments.get("url")
        query = arguments.get("query")
        if url is None:
            return "URL is none"
        if query is None:
            return "Query is none"
        if not isinstance(url, str) or not isinstance(query, str):
            return "URL and query must be strings"
        if _is_url_excluded(url, self._exclude_domains):
            return "URL is in excluded domains"

        payload = {
            "urls": url,
            "query": query,
            "extract_depth": self._config["extract_depth"],
            "format": self._config["extract_format"],
        }
        results = await self._tracked_post_json("/extract", payload, request_id=request_id, function="extract")
        if results.get("error"):
            return results["error"]

        raw_content = _first_raw_content(results)
        if not raw_content:
            return "No content found."

        domain = _extract_domain(url)
        cleaned = _clean_text(raw_content)
        truncated, was_truncated = _truncate_text(cleaned, int(self._config["max_result_chars"]))
        numbered = _add_line_numbers(truncated)
        header = f'Content from: {domain}\nURL: {url}\nQuery: "{query}"\n========================================\n'
        footer = "\n[...truncated, use scroll_page for full content]" if was_truncated else ""
        return header + numbered + footer

    async def _scroll_page(self, arguments: dict[str, Any], *, request_id: str | None) -> str:
        url = arguments.get("url")
        if url is None:
            return "URL is none"
        if not isinstance(url, str):
            return "URL must be a string"
        if _is_url_excluded(url, self._exclude_domains):
            return "URL is in excluded domains"

        start_index = int(arguments.get("start_index", 0) or 0)
        n = int(arguments.get("n", self._config["scroll_words"]) or self._config["scroll_words"])
        start_index = max(0, start_index)
        n = max(1, n)

        if url in self._page_cache:
            page_content = self._page_cache[url]
        else:
            payload = {
                "urls": url,
                "extract_depth": self._config["extract_depth"],
                "format": self._config["extract_format"],
            }
            results = await self._tracked_post_json("/extract", payload, request_id=request_id, function="extract")
            if results.get("error"):
                return results["error"]
            page_content = _first_raw_content(results)
            self._page_cache[url] = page_content

        words = page_content.split()
        total_words = len(words)
        sliced_words = words[start_index : start_index + n]
        chunk_text = " ".join(sliced_words)
        domain = _extract_domain(url)
        cleaned = _clean_text(chunk_text)
        numbered = _add_line_numbers(cleaned)
        end_index = min(start_index + n, total_words)
        header = (
            f"Page content from: {domain}\n"
            f"URL: {url}\n"
            f"Showing words [{start_index}-{end_index}] of {total_words}\n"
            f"========================================\n"
        )
        return header + numbered


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
