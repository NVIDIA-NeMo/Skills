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

"""Wikipedia MCP tool — direct REST + MediaWiki Action API.

Replaces the previous python-`wikipedia` wrapper which produced ~22% junk
responses (HTTP-429 disguised as JSON-decode errors, "page not found",
"ambiguous"). Wikimedia REST and Action APIs accept a much higher request
volume (hundreds of req/sec) **provided** the client identifies itself
with a real `User-Agent` — we set one explicitly per their policy:
https://meta.wikimedia.org/wiki/User-Agent_policy

Tool surface (kept stable for backward compatibility):
- `wikipedia-search(query, num_results)` — search hits with snippets
- `wikipedia-page(title)`              — full plain-text extract (truncated)

New auxiliary:
- `wikipedia-section(title, section)`  — fetch a single named section

Usage:
    ++tool_modules=[nemo_skills.mcp.servers.web.wikipedia_tool::WikipediaSearchTool]
"""

import asyncio
import fcntl
import hashlib
import html
import json
import logging
import os
import re
import tempfile
import time
from typing import Annotated, Any

import httpx
from pydantic import Field

from nemo_skills.mcp.tool_manager import Tool

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
WIKI_LANG = os.getenv("WIKIPEDIA_LANG", "en")
ACTION_BASE = f"https://{WIKI_LANG}.wikipedia.org/w/api.php"
REST_BASE = f"https://{WIKI_LANG}.wikipedia.org/api/rest_v1"

CONTACT = os.getenv("WIKIPEDIA_CONTACT", "nemo-skills-research@nvidia.com")
USER_AGENT = f"NeMo-Skills/MCP ({CONTACT})"

EXTRACT_LIMIT = 2500  # chars of full-page/section extract returned to the model
SNIPPET_LIMIT = 250  # chars per search-result snippet
HTTP_TIMEOUT = 20.0
NUM_RETRIES = 3
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 15.0
CACHE_MAX_SIZE = 512
MAX_FACTS = 8
REQUEST_INTERVAL = float(os.getenv("WIKIPEDIA_REQUEST_INTERVAL", "1.0"))
RATE_LIMIT_LOCK = os.getenv(
    "WIKIPEDIA_RATE_LIMIT_LOCK",
    os.path.join(tempfile.gettempdir(), "nemo_skills_wikipedia_api_rate_limit.lock"),
)
EXPECTED_HTTP_ERRORS = (httpx.HTTPError, RuntimeError, json.JSONDecodeError)

_cache: dict[str, str] = {}
_request_lock = asyncio.Lock()


# ── Helpers ────────────────────────────────────────────────────────────────


def _cache_key(*args: Any) -> str:
    """Build a stable cache key for tool arguments."""
    return hashlib.sha256(json.dumps(args, default=str, sort_keys=True).encode()).hexdigest()


def _cache_get(key: str) -> str | None:
    """Return a cached response, if present."""
    return _cache.get(key)


def _cache_set(key: str, value: str) -> None:
    """Store a response with simple bounded FIFO eviction."""
    if len(_cache) >= CACHE_MAX_SIZE:
        _cache.pop(next(iter(_cache)))
    _cache[key] = value


def _strip_html(s: str) -> str:
    """Strip HTML tags and decode entities (used for action-API search snippets)."""
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s).strip()


def _truncate(text: str, limit: int) -> str:
    """Trim text to a word boundary within a character budget."""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " …[truncated]"


def _page_url(title: str) -> str:
    """Return the canonical Wikipedia page URL for a title."""
    return f"https://{WIKI_LANG}.wikipedia.org/wiki/{title.replace(' ', '_')}"


def _sentence_split(text: str) -> list[str]:
    """Small sentence splitter good enough for compact fact snippets."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


async def _page_extract(title: str) -> tuple[str, str, str] | tuple[None, None, str]:
    """Fetch clean article text. Returns (rendered_title, url, extract) or error."""
    t = title.strip()
    if not t:
        return None, None, "Wikipedia lookup failed: empty title."
    params = {
        "action": "query",
        "prop": "extracts|info",
        "explaintext": 1,
        "redirects": 1,
        "inprop": "url",
        "titles": t,
        "format": "json",
        "formatversion": "2",
    }
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    async with httpx.AsyncClient(headers=headers) as client:
        data = await _http_get_json(client, ACTION_BASE, params=params)
    pages = (data.get("query") or {}).get("pages") or []
    if not pages:
        return None, None, f"Wikipedia page {t!r} not found."
    page = pages[0]
    if page.get("missing"):
        suggest = await _suggest_titles(t)
        msg = f"Wikipedia page {t!r} not found."
        if suggest:
            msg += f"\nDid you mean one of: {', '.join(suggest)}?"
        return None, None, msg
    rendered_title = page.get("title", t)
    url = page.get("fullurl") or _page_url(rendered_title)
    extract = page.get("extract") or ""
    return rendered_title, url, extract


async def _http_get_json(client: httpx.AsyncClient, url: str, params: dict[str, Any] | None) -> Any:
    """GET JSON with retry on 429/5xx and exponential backoff."""
    delay = INITIAL_BACKOFF
    last_err: Exception | None = None
    for attempt in range(NUM_RETRIES + 1):
        try:
            await _rate_limit()
            r = await client.get(url, params=params, timeout=HTTP_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                if attempt < NUM_RETRIES:
                    retry_after = _retry_after_seconds(r)
                    await asyncio.sleep(retry_after if retry_after is not None else delay)
                    delay = min(delay * 2, MAX_BACKOFF)
                    continue
                raise last_err
            r.raise_for_status()
            return r.json()
        except (httpx.RequestError, httpx.TimeoutException) as e:
            last_err = e
            if attempt < NUM_RETRIES:
                await asyncio.sleep(delay)
                delay = min(delay * 2, MAX_BACKOFF)
                continue
            raise
    raise last_err or RuntimeError("unknown error")


def _retry_after_seconds(response: httpx.Response) -> float | None:
    """Parse a bounded Retry-After delay from an HTTP response."""
    raw = response.headers.get("retry-after")
    if not raw:
        return None
    try:
        return max(0.0, min(float(raw), MAX_BACKOFF))
    except ValueError:
        return None


async def _rate_limit() -> None:
    """Throttle Wikimedia API requests across concurrent chunk workers.

    Multiple Slurm chunks can start on different nodes and otherwise stampede
    the public API. A shared Lustre lock serializes requests across processes.
    """
    if REQUEST_INTERVAL <= 0:
        return

    async with _request_lock:
        lock_path = RATE_LIMIT_LOCK
        try:
            os.makedirs(os.path.dirname(lock_path), exist_ok=True)
            with open(lock_path, "a+", encoding="utf-8") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.seek(0)
                raw = f.read().strip()
                last = float(raw) if raw else 0.0
                now = time.time()
                wait = REQUEST_INTERVAL - (now - last)
                if wait > 0:
                    await asyncio.sleep(wait)
                    now = time.time()
                f.seek(0)
                f.truncate()
                f.write(str(now))
                f.flush()
                fcntl.flock(f, fcntl.LOCK_UN)
        except OSError:
            # Fall back to per-process throttling if the shared mount is missing.
            await asyncio.sleep(REQUEST_INTERVAL)


# ── Tools ──────────────────────────────────────────────────────────────────


async def wikipedia_search(
    query: Annotated[str, Field(description="Search query for Wikipedia articles.")],
    num_results: Annotated[int, Field(description="Number of search results to return (1-5).")] = 3,
) -> str:
    """Search Wikipedia for articles. Returns titles, URLs, and ~400-char snippets.

    Backed by the MediaWiki Action API (`action=query&list=search`), which
    is the canonical full-text search endpoint and accepts hundreds of
    req/sec when the client identifies itself with a User-Agent.
    """
    # Keep search output compact. The model can drill into a specific page or
    # section after seeing the title list.
    n = int(num_results or 3)
    if n < 1 or n > 5:
        return "num_results must be between 1 and 5."
    cache_key = _cache_key("search", query, n)
    if cached := _cache_get(cache_key):
        return cached

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": n,
        "srprop": "snippet|titlesnippet|sectionsnippet",
        "format": "json",
        "formatversion": "2",
    }
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    async with httpx.AsyncClient(headers=headers) as client:
        try:
            data = await _http_get_json(client, ACTION_BASE, params=params)
        except EXPECTED_HTTP_ERRORS as e:
            return f"Wikipedia search failed: {e}"

    hits = (data.get("query") or {}).get("search") or []
    if not hits:
        return f"No Wikipedia articles found for query: {query!r}."

    blocks: list[str] = []
    for h in hits:
        title = h.get("title", "(untitled)")
        snippet = _strip_html(h.get("snippet", ""))
        section = _strip_html(h.get("sectiontitle", ""))
        snippet = _truncate(snippet, SNIPPET_LIMIT)
        line = f"**{title}**\nURL: {_page_url(title)}"
        if section:
            line += f"\nSection: {section}"
        line += f"\n{snippet}" if snippet else ""
        blocks.append(line)
    result = "\n\n---\n\n".join(blocks)
    _cache_set(cache_key, result)
    return result


async def wikipedia_page(
    title: Annotated[
        str,
        Field(description="Title of the Wikipedia article (case-insensitive; spaces or underscores OK)."),
    ],
) -> str:
    """Fetch the plain-text extract of a Wikipedia article (up to ~4000 chars).

    Uses the MediaWiki Action API with `prop=extracts&explaintext=1`, which
    returns the article body as clean UTF-8 text (no HTML, no wiki markup)
    and follows redirects automatically. Truncated to fit the model context
    budget; use `wikipedia-section` to drill into a specific section.
    """
    t = title.strip()
    if not t:
        return "Wikipedia lookup failed: empty title."

    cache_key = _cache_key("page", t.lower())
    if cached := _cache_get(cache_key):
        return cached

    params = {
        "action": "query",
        "prop": "extracts|info",
        "explaintext": 1,
        "redirects": 1,
        "inprop": "url",
        "titles": t,
        "format": "json",
        "formatversion": "2",
    }
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    async with httpx.AsyncClient(headers=headers) as client:
        try:
            data = await _http_get_json(client, ACTION_BASE, params=params)
        except EXPECTED_HTTP_ERRORS as e:
            return f"Wikipedia lookup failed: {e}"

    pages = (data.get("query") or {}).get("pages") or []
    if not pages:
        return f"Wikipedia page {t!r} not found."
    page = pages[0]
    if page.get("missing"):
        # Try a fuzzy fallback via the search endpoint to suggest alternatives.
        suggest = await _suggest_titles(t)
        msg = f"Wikipedia page {t!r} not found."
        if suggest:
            msg += f"\nDid you mean one of: {', '.join(suggest)}?"
        return msg
    extract = page.get("extract") or ""
    rendered_title = page.get("title", t)
    url = page.get("fullurl") or _page_url(rendered_title)
    if not extract:
        return f"**{rendered_title}**\nURL: {url}\n(No plain-text extract available — page may be a disambiguation or stub.)"
    body = _truncate(extract, EXTRACT_LIMIT)
    rendered = f"**{rendered_title}**\nURL: {url}\n\n{body}"
    _cache_set(cache_key, rendered)
    return rendered


async def wikipedia_summary(
    title: Annotated[str, Field(description="Title of the Wikipedia article.")],
    max_chars: Annotated[int, Field(description="Maximum characters to return (200-1500).")] = 700,
) -> str:
    """Fetch only the lead summary of an article.

    This is the preferred first retrieval step after `wikipedia-search`: it is
    much cheaper than `wikipedia-page` and usually enough for factual grounding.
    """
    t = title.strip()
    max_chars = max(200, min(int(max_chars or 700), 1500))
    cache_key = _cache_key("summary", t.lower(), max_chars)
    if cached := _cache_get(cache_key):
        return cached

    params = {
        "action": "query",
        "prop": "extracts|info",
        "exintro": 1,
        "explaintext": 1,
        "redirects": 1,
        "inprop": "url",
        "titles": t,
        "format": "json",
        "formatversion": "2",
    }
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    async with httpx.AsyncClient(headers=headers) as client:
        try:
            data = await _http_get_json(client, ACTION_BASE, params=params)
        except EXPECTED_HTTP_ERRORS as e:
            return f"Wikipedia summary failed: {e}"
    pages = (data.get("query") or {}).get("pages") or []
    if not pages or pages[0].get("missing"):
        return f"Wikipedia page {t!r} not found."
    page = pages[0]
    rendered_title = page.get("title", t)
    url = page.get("fullurl") or _page_url(rendered_title)
    extract = _truncate(page.get("extract") or "", max_chars)
    rendered = f"**{rendered_title}**\nURL: {url}\n\n{extract}"
    _cache_set(cache_key, rendered)
    return rendered


async def wikipedia_sections(
    title: Annotated[str, Field(description="Title of the Wikipedia article.")],
    max_sections: Annotated[int, Field(description="Maximum number of section headings to list (1-50).")] = 25,
) -> str:
    """List article section headings without returning section bodies.

    Use this before `wikipedia-section` when you need a specific part of a
    long article. This mirrors mature Wikipedia MCPs that expose
    section-first retrieval to avoid dumping whole pages into context.
    """
    t = title.strip()
    max_sections = max(1, min(int(max_sections or 25), 50))
    cache_key = _cache_key("sections-list", t.lower(), max_sections)
    if cached := _cache_get(cache_key):
        return cached
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    async with httpx.AsyncClient(headers=headers) as client:
        try:
            data = await _http_get_json(
                client,
                ACTION_BASE,
                params={
                    "action": "parse",
                    "page": t,
                    "prop": "sections",
                    "redirects": 1,
                    "format": "json",
                    "formatversion": "2",
                },
            )
        except EXPECTED_HTTP_ERRORS as e:
            return f"Wikipedia section listing failed: {e}"
    sections = (data.get("parse") or {}).get("sections") or []
    if not sections:
        return f"Wikipedia page {t!r} has no parsed sections."
    lines = [f"**{t}** sections (use wikipedia-section with one heading):"]
    for sec in sections[:max_sections]:
        level = sec.get("toclevel") or sec.get("level") or "?"
        lines.append(f"- L{level} | {sec.get('line', '')} | anchor={sec.get('anchor', '')}")
    if len(sections) > max_sections:
        lines.append(f"... {len(sections) - max_sections} more sections omitted")
    rendered = "\n".join(lines)
    _cache_set(cache_key, rendered)
    return rendered


async def wikipedia_query_summary(
    title: Annotated[str, Field(description="Title of the Wikipedia article.")],
    query: Annotated[str, Field(description="Term or phrase to locate within the article.")],
    max_chars: Annotated[int, Field(description="Maximum characters to return (200-1500).")] = 700,
) -> str:
    """Return a bounded snippet around a query inside an article."""
    t = title.strip()
    q = query.strip()
    max_chars = max(200, min(int(max_chars or 700), 1500))
    cache_key = _cache_key("query-summary", t.lower(), q.lower(), max_chars)
    if cached := _cache_get(cache_key):
        return cached
    try:
        rendered_title, url, extract = await _page_extract(t)
    except EXPECTED_HTTP_ERRORS as e:
        return f"Wikipedia query summary failed: {e}"
    if rendered_title is None:
        return extract
    hay = extract or ""
    idx = hay.lower().find(q.lower()) if q else -1
    if idx < 0:
        snippet = _truncate(hay, max_chars)
        note = f"Query {q!r} not found exactly; returning lead text."
    else:
        start = max(0, idx - max_chars // 2)
        end = min(len(hay), start + max_chars)
        snippet = hay[start:end].strip()
        if start > 0:
            snippet = "..." + snippet
        if end < len(hay):
            snippet += "..."
        note = f"Snippet around query {q!r}."
    rendered = f"**{rendered_title}**\nURL: {url}\n{note}\n\n{snippet}"
    _cache_set(cache_key, rendered)
    return rendered


async def wikipedia_key_facts(
    title: Annotated[str, Field(description="Title of the Wikipedia article.")],
    topic: Annotated[str, Field(description="Optional topic to focus facts on.")] = "",
    count: Annotated[int, Field(description="Number of facts to return (1-8).")] = 5,
) -> str:
    """Extract a few compact factual sentences from an article.

    This is deliberately simple and deterministic: it selects sentences from
    the article or a query-focused window, not an LLM-generated summary.
    """
    t = title.strip()
    topic = (topic or "").strip()
    count = max(1, min(int(count or 5), MAX_FACTS))
    cache_key = _cache_key("key-facts", t.lower(), topic.lower(), count)
    if cached := _cache_get(cache_key):
        return cached
    try:
        rendered_title, url, extract = await _page_extract(t)
    except EXPECTED_HTTP_ERRORS as e:
        return f"Wikipedia key facts failed: {e}"
    if rendered_title is None:
        return extract
    text = extract or ""
    if topic:
        idx = text.lower().find(topic.lower())
        if idx >= 0:
            text = text[max(0, idx - 1000) : idx + 2500]
    facts = _sentence_split(text)[:count]
    if not facts:
        return f"No extractable facts found for {t!r}."
    rendered = f"**{rendered_title}**\nURL: {url}\n" + "\n".join(f"- {f}" for f in facts)
    _cache_set(cache_key, rendered)
    return rendered


async def wikipedia_section(
    title: Annotated[str, Field(description="Title of the Wikipedia article.")],
    section: Annotated[
        str,
        Field(description="Section heading (e.g. 'History', 'Applications'). Case-insensitive substring match."),
    ],
) -> str:
    """Fetch one named section of a Wikipedia article.

    First lists the article's sections via `prop=sections`, picks the one
    whose heading best matches the requested name, and fetches that
    section's plain-text content via `section=N`. Useful when the full
    article exceeds the context budget but only one section is relevant.
    """
    t = title.strip()
    s_query = section.strip().lower()
    if not t or not s_query:
        return "Wikipedia section lookup failed: title or section is empty."

    cache_key = _cache_key("section", t.lower(), s_query)
    if cached := _cache_get(cache_key):
        return cached

    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    async with httpx.AsyncClient(headers=headers) as client:
        # 1) list sections
        try:
            sec_data = await _http_get_json(
                client,
                ACTION_BASE,
                params={
                    "action": "parse",
                    "page": t,
                    "prop": "sections",
                    "redirects": 1,
                    "format": "json",
                    "formatversion": "2",
                },
            )
        except EXPECTED_HTTP_ERRORS as e:
            return f"Wikipedia section listing failed: {e}"
        sections = (sec_data.get("parse") or {}).get("sections") or []
        if not sections:
            return f"Wikipedia page {t!r} has no parsed sections."
        # find best section by substring match on heading (line)
        best = None
        for sec in sections:
            line = (sec.get("line") or "").lower()
            if s_query in line:
                best = sec
                break
        if best is None:
            available = ", ".join(s.get("line", "") for s in sections[:15])
            return f"Section matching {section!r} not found in {t!r}. Available: {available}"

        # 2) fetch that section's text
        try:
            text_data = await _http_get_json(
                client,
                ACTION_BASE,
                params={
                    "action": "parse",
                    "page": t,
                    "section": best.get("index"),
                    "prop": "wikitext|text",
                    "redirects": 1,
                    "format": "json",
                    "formatversion": "2",
                },
            )
        except EXPECTED_HTTP_ERRORS as e:
            return f"Wikipedia section fetch failed: {e}"

        # Prefer the rendered HTML stripped (cleaner than raw wikitext).
        html_blob = (text_data.get("parse") or {}).get("text") or ""
        if isinstance(html_blob, dict):
            html_blob = html_blob.get("*", "")
        body = _strip_html(html_blob)
        if not body:
            body = (text_data.get("parse") or {}).get("wikitext", "")
            if isinstance(body, dict):
                body = body.get("*", "")
    rendered = f"**{t}** › {best.get('line')}\nURL: {_page_url(t)}#{best.get('anchor', '')}\n\n{_truncate(body, EXTRACT_LIMIT)}"
    _cache_set(cache_key, rendered)
    return rendered


async def _suggest_titles(query: str, n: int = 5) -> list[str]:
    """Best-effort: ask MediaWiki search for likely-intended titles."""
    params = {
        "action": "opensearch",
        "search": query,
        "limit": n,
        "namespace": 0,
        "format": "json",
    }
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    try:
        async with httpx.AsyncClient(headers=headers) as client:
            r = await client.get(ACTION_BASE, params=params, timeout=HTTP_TIMEOUT)
            if r.status_code != 200:
                return []
            data = r.json()
            return data[1] if isinstance(data, list) and len(data) > 1 else []
    except EXPECTED_HTTP_ERRORS:
        return []


# ── Tool provider ──────────────────────────────────────────────────────────


class WikipediaSearchTool(Tool):
    """Direct Wikipedia search/retrieval tool."""

    def __init__(self) -> None:
        self._config: dict[str, Any] = {
            "num_results": 3,
            "max_chars": EXTRACT_LIMIT,
            "max_sections": 40,
            "count": 5,
        }

    def default_config(self) -> dict[str, Any]:
        return dict(self._config)

    def configure(self, overrides: dict[str, Any] | None = None, context: dict[str, Any] | None = None) -> None:
        if not overrides:
            return

        unknown = set(overrides) - set(self._config)
        if unknown:
            raise ValueError(f"Unknown WikipediaSearchTool override(s): {sorted(unknown)}")

        if "num_results" in overrides:
            num_results = int(overrides["num_results"])
            if num_results < 1 or num_results > 5:
                raise ValueError("num_results must be between 1 and 5.")
            self._config["num_results"] = num_results
        if "max_chars" in overrides:
            self._config["max_chars"] = max(200, min(int(overrides["max_chars"]), 1500))
        if "max_sections" in overrides:
            self._config["max_sections"] = max(1, min(int(overrides["max_sections"]), 50))
        if "count" in overrides:
            self._config["count"] = max(1, min(int(overrides["count"]), MAX_FACTS))

    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "wikipedia-search",
                "description": "Search Wikipedia articles and return compact snippets.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Search query."}},
                    "required": ["query"],
                },
            },
            {
                "name": "wikipedia-page",
                "description": "Fetch a Wikipedia article extract by title.",
                "input_schema": {
                    "type": "object",
                    "properties": {"title": {"type": "string", "description": "Article title."}},
                    "required": ["title"],
                },
            },
            {
                "name": "wikipedia-summary",
                "description": "Fetch a compact Wikipedia article summary by title.",
                "input_schema": {
                    "type": "object",
                    "properties": {"title": {"type": "string", "description": "Article title."}},
                    "required": ["title"],
                },
            },
            {
                "name": "wikipedia-sections",
                "description": "List section headings for a Wikipedia page.",
                "input_schema": {
                    "type": "object",
                    "properties": {"title": {"type": "string", "description": "Article title."}},
                    "required": ["title"],
                },
            },
            {
                "name": "wikipedia-section",
                "description": "Fetch one named Wikipedia page section.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Article title."},
                        "section": {"type": "string", "description": "Section heading."},
                    },
                    "required": ["title", "section"],
                },
            },
            {
                "name": "wikipedia-query-summary",
                "description": "Return a compact snippet around a query inside a Wikipedia article.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "Article title."},
                        "query": {"type": "string", "description": "Term or phrase to locate within the article."},
                    },
                    "required": ["title", "query"],
                },
            },
            {
                "name": "wikipedia-key-facts",
                "description": "Extract compact key facts from a Wikipedia page.",
                "input_schema": {
                    "type": "object",
                    "properties": {"title": {"type": "string", "description": "Article title."}},
                    "required": ["title"],
                },
            },
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any], extra_args: dict[str, Any] | None = None):
        arguments = dict(arguments or {})
        if tool_name == "wikipedia-search":
            arguments.setdefault("num_results", self._config["num_results"])
            return await wikipedia_search(**arguments)
        if tool_name == "wikipedia-page":
            return await wikipedia_page(**arguments)
        if tool_name == "wikipedia-summary":
            arguments.setdefault("max_chars", self._config["max_chars"])
            return await wikipedia_summary(**arguments)
        if tool_name == "wikipedia-sections":
            arguments.setdefault("max_sections", self._config["max_sections"])
            return await wikipedia_sections(**arguments)
        if tool_name == "wikipedia-section":
            return await wikipedia_section(**arguments)
        if tool_name == "wikipedia-query-summary":
            arguments.setdefault("max_chars", self._config["max_chars"])
            return await wikipedia_query_summary(**arguments)
        if tool_name == "wikipedia-key-facts":
            arguments.setdefault("count", self._config["count"])
            return await wikipedia_key_facts(**arguments)
        return f"Error: unknown tool '{tool_name}'"
