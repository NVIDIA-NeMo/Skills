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

"""ArXiv search MCP tool for scientific paper retrieval.

Runs outside the sandbox (no network blocking). Uses the free arXiv API
via the `arxiv` pip package. No API key required.

Prerequisites:
    pip install arxiv

Usage:
    ++tool_modules=[nemo_skills.mcp.servers.arxiv_tool::ArxivSearchTool]
"""

import hashlib
import json
import logging
import time
from threading import Lock
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from nemo_skills.mcp.tool_providers import MCPClientTool

logger = logging.getLogger(__name__)

mcp = FastMCP(name="arxiv")

MAX_RESULTS = 10
_REQUEST_INTERVAL = 3.0
_NUM_RETRIES = 5
_INITIAL_DELAY = 5.0
_MAX_DELAY = 60.0
_CACHE_MAX_SIZE = 256

_last_request_time = 0.0
_rate_lock = Lock()
_cache: dict[str, str] = {}


def _rate_limit():
    """Enforce minimum 3-second gap between ArXiv API calls."""
    global _last_request_time
    with _rate_lock:
        now = time.monotonic()
        wait = _REQUEST_INTERVAL - (now - _last_request_time)
        if wait > 0:
            time.sleep(wait)
        _last_request_time = time.monotonic()


def _cache_key(*args) -> str:
    return hashlib.sha256(json.dumps(args, sort_keys=True).encode()).hexdigest()


def _with_retry(fn):
    """Execute fn with exponential backoff. Rate-limits each attempt."""
    delay = _INITIAL_DELAY
    for attempt in range(_NUM_RETRIES + 1):
        try:
            _rate_limit()
            return fn()
        except Exception as e:
            if attempt == _NUM_RETRIES:
                raise
            logger.warning(
                "ArXiv attempt %d/%d failed: %s — retrying in %.0fs",
                attempt + 1,
                _NUM_RETRIES + 1,
                e,
                delay,
            )
            time.sleep(delay)
            delay = min(delay * 2, _MAX_DELAY)


@mcp.tool(name="arxiv-search")
def arxiv_search(
    query: Annotated[
        str, Field(description="Search query for arXiv papers (supports arXiv query syntax: au:, ti:, abs:, cat:).")
    ],
    max_results: Annotated[int, Field(description="Maximum number of results to return.")] = 3,
) -> str:
    """Search arXiv for scientific papers. Returns titles, abstracts, and URLs."""
    import arxiv

    if max_results > MAX_RESULTS:
        max_results = MAX_RESULTS

    key = _cache_key("search", query, max_results)
    if key in _cache:
        return _cache[key]

    def _fetch():
        client = arxiv.Client(page_size=max_results, num_retries=1, delay_seconds=0)
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = []
        for paper in client.results(search):
            results.append(
                f"**{paper.title}**\n"
                f"Authors: {', '.join(a.name for a in paper.authors[:5])}"
                f"{'...' if len(paper.authors) > 5 else ''}\n"
                f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
                f"URL: {paper.entry_id}\n"
                f"Abstract: {paper.summary[:500]}{'...' if len(paper.summary) > 500 else ''}\n"
            )
        return results

    try:
        results = _with_retry(_fetch)
        if not results:
            return "No papers found for this query."
        result_str = "\n---\n".join(results)
        if len(_cache) < _CACHE_MAX_SIZE:
            _cache[key] = result_str
        return result_str
    except Exception as e:
        return f"ArXiv search failed: {e}"


@mcp.tool(name="arxiv-get")
def arxiv_get(
    paper_id: Annotated[str, Field(description="arXiv paper ID (e.g. '2301.07041' or '2301.07041v1').")],
) -> str:
    """Fetch a specific arXiv paper by ID. Returns full title, authors, abstract, and metadata."""
    import arxiv

    key = _cache_key("get", paper_id)
    if key in _cache:
        return _cache[key]

    def _fetch():
        client = arxiv.Client(page_size=1, num_retries=1, delay_seconds=0)
        search = arxiv.Search(id_list=[paper_id])
        return next(client.results(search), None)

    try:
        paper = _with_retry(_fetch)
        if paper is None:
            return f"Paper {paper_id} not found on arXiv."
        result_str = (
            f"**{paper.title}**\n"
            f"Authors: {', '.join(a.name for a in paper.authors)}\n"
            f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
            f"Updated: {paper.updated.strftime('%Y-%m-%d')}\n"
            f"Categories: {', '.join(paper.categories)}\n"
            f"URL: {paper.entry_id}\n"
            f"PDF: {paper.pdf_url}\n\n"
            f"Abstract:\n{paper.summary}"
        )
        if len(_cache) < _CACHE_MAX_SIZE:
            _cache[key] = result_str
        return result_str
    except Exception as e:
        return f"ArXiv lookup failed: {e}"


class ArxivSearchTool(MCPClientTool):
    def __init__(self) -> None:
        super().__init__()
        self.apply_config_updates(
            {
                "client": "nemo_skills.mcp.clients.MCPStdioClient",
                "client_params": {
                    "command": "python",
                    "args": ["-m", "nemo_skills.mcp.servers.arxiv_tool"],
                },
                "hide_args": {
                    "arxiv-search": ["max_results"],
                },
            }
        )


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
