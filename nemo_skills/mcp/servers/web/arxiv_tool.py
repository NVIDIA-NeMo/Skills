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

"""Scientific paper search MCP tool.

Provides search and lookup over the arXiv preprint corpus. The model-facing
search path uses arXiv's native Atom API with a strict per-process request
delay so it does not depend on OpenAlex budgets/keys. OpenAlex is retained
only for `arxiv-get` metadata lookups by DOI/OpenAlex/PMID.

Tool names are kept as `arxiv-search` / `arxiv-get` for backward
compatibility with prior generation runs that already learned to use them.

Usage:
    ++tool_modules=[nemo_skills.mcp.servers.web.arxiv_tool::ArxivSearchTool]
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from collections import OrderedDict
from html import unescape
from html.parser import HTMLParser
from typing import Annotated, Any

import httpx
from pydantic import Field

from nemo_skills.mcp.tool_manager import Tool

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
OPENALEX_BASE = "https://api.openalex.org"
ARXIV_BASE = "https://export.arxiv.org/api/query"

# Polite-pool email is enough to lift OpenAlex rate limits even without a key.
POLITE_EMAIL = os.getenv("OPENALEX_EMAIL", "nemo-skills-research@nvidia.com")
USER_AGENT = f"NeMo-Skills/MCP ({POLITE_EMAIL})"

# Premium tier API key — lifts the ~1k/day search-query free-tier limit.
# If the key's daily budget is exhausted (HTTP 429 "Insufficient budget"),
# we automatically drop to the polite-pool tier for the rest of the process.
_openalex_api_key: str | None = (os.getenv("OPENALEX_API_KEY") or "").strip() or None
_key_exhausted = False

MAX_RESULTS = 10
ABSTRACT_LIMIT = 3000  # chars per abstract returned to the model
PAPER_CHUNK_LIMIT = 3500
HTTP_TIMEOUT = 30.0
NUM_RETRIES = 3
INITIAL_BACKOFF = 1.0
MAX_BACKOFF = 30.0
CACHE_MAX_SIZE = 512
PAPER_CACHE_MAX_SIZE = 32

# arXiv's public API asks clients to wait 3 seconds between requests. The
# evaluation can launch many chunk processes, often on different nodes, so a
# process-local lock is not enough. Use a small shared lock file on the
# workspace mount when available; fall back to /tmp for local smoke tests.
ARXIV_REQUEST_INTERVAL = float(os.getenv("ARXIV_REQUEST_INTERVAL", "4.0"))
ARXIV_RATE_LIMIT_LOCK = os.getenv(
    "ARXIV_RATE_LIMIT_LOCK",
    os.path.join(tempfile.gettempdir(), "nemo_skills_arxiv_api_rate_limit.lock"),
)
_arxiv_last_request = 0.0
_arxiv_lock = asyncio.Lock()

_cache: dict[str, str] = {}
_paper_cache: OrderedDict[str, tuple[str, str]] = OrderedDict()


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
        # Drop one arbitrary entry; simple FIFO-ish eviction.
        _cache.pop(next(iter(_cache)))
    _cache[key] = value


def _paper_cache_get(key: str) -> tuple[str, str] | None:
    """Return cached paper text and refresh recency."""
    value = _paper_cache.get(key)
    if value is not None:
        _paper_cache.move_to_end(key)
    return value


def _paper_cache_set(key: str, value: tuple[str, str]) -> None:
    """Store full paper text in a bounded LRU cache."""
    _paper_cache[key] = value
    _paper_cache.move_to_end(key)
    while len(_paper_cache) > PAPER_CACHE_MAX_SIZE:
        _paper_cache.popitem(last=False)


def _reconstruct_abstract(inv_idx: dict[str, list[int]] | None) -> str:
    """Reconstruct OpenAlex inverted-index abstract back to plain text."""
    if not inv_idx:
        return ""
    positions: dict[int, str] = {}
    for word, indices in inv_idx.items():
        for idx in indices:
            positions[idx] = word
    if not positions:
        return ""
    return " ".join(positions[i] for i in sorted(positions))


def _truncate(text: str, limit: int = ABSTRACT_LIMIT) -> str:
    """Trim text to a word boundary within a character budget."""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0] + " …[truncated]"


class _ArxivHTMLTextParser(HTMLParser):
    """Tiny HTML-to-text extractor that preserves headings for section lookup."""

    _SKIP_TAGS = {"script", "style", "noscript", "svg", "math"}
    _BLOCK_TAGS = {"p", "div", "section", "article", "li", "br", "tr"}
    _HEADING_TAGS = {"h1", "h2", "h3", "h4"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0
        self._heading_tag: str | None = None
        self._heading_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return
        if self._skip_depth:
            return
        if tag in self._HEADING_TAGS:
            self._heading_tag = tag
            self._heading_buf = []
            return
        if tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self._SKIP_TAGS and self._skip_depth:
            self._skip_depth -= 1
            return
        if self._skip_depth:
            return
        if tag == self._heading_tag:
            heading = " ".join("".join(self._heading_buf).split())
            if heading:
                level = int(tag[1])
                self.parts.append("\n\n" + ("#" * level) + f" {heading}\n")
            self._heading_tag = None
            self._heading_buf = []
        elif tag in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._heading_tag:
            self._heading_buf.append(data)
            return
        text = unescape(data)
        if text.strip():
            self.parts.append(text)

    def text(self) -> str:
        raw = "".join(self.parts)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def _normalize_id(paper_id: str) -> str:
    """Map common ID forms to OpenAlex-compatible format."""
    pid = paper_id.strip()
    low = pid.lower()
    if low.startswith(("https://openalex.org/", "openalex:")):
        return pid.split("/")[-1].split(":")[-1]
    if low.startswith("doi:") or low.startswith("https://doi.org/") or low.startswith("http://doi.org/"):
        doi = re.split(r"(?i)https?://doi\.org/", pid, maxsplit=1)[-1]
        doi = re.sub(r"(?i)^doi:\s*", "", doi).strip()
        return f"doi:{doi}"
    if low.startswith("10.") and "/" in pid:
        return f"doi:{pid}"
    if low.startswith("arxiv:"):
        arx = pid[6:].strip()
        return f"doi:10.48550/arXiv.{arx}"
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", pid):  # bare arXiv id like 2103.15348
        return f"doi:10.48550/arXiv.{pid.split('v')[0]}"
    if low.startswith("pmid:"):
        return pid
    if pid.startswith("W") and pid[1:].isdigit():
        return pid
    return pid  # let OpenAlex try as-is


def _extract_arxiv_id(paper_id: str) -> str | None:
    """Extract a bare arXiv id from common user inputs."""
    pid = paper_id.strip()
    low = pid.lower()
    if low.startswith("arxiv:"):
        pid = pid[6:].strip()
    if "arxiv.org/abs/" in low or "arxiv.org/pdf/" in low or "arxiv.org/html/" in low:
        pid = re.split(r"arxiv\.org/(?:abs|pdf|html)/", low, maxsplit=1)[-1]
        pid = pid.split("?")[0].split("#")[0].replace(".pdf", "")
    if "10.48550/arxiv." in low:
        pid = re.split(r"10\.48550/arxiv\.", low, flags=re.I, maxsplit=1)[-1]
    pid = pid.strip()
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", pid) or re.match(r"^[a-z-]+(?:\.[A-Z]{2})?/\d{7}(v\d+)?$", pid, re.I):
        return pid
    return None


async def _fetch_paper_text(paper_id: str) -> tuple[str, str]:
    """Fetch arXiv full text via native HTML, with ar5iv fallback."""
    arxiv_id = _extract_arxiv_id(paper_id)
    if not arxiv_id:
        raise ValueError(
            "Full-text chunking needs an arXiv id (e.g. '2103.15348' or 'arXiv:2103.15348'). "
            "Use arxiv-get for DOI/OpenAlex metadata lookups."
        )
    cache_key = arxiv_id.split("v")[0]
    cached_paper = _paper_cache_get(cache_key)
    if cached_paper is not None:
        return cached_paper

    urls = [
        f"https://arxiv.org/html/{arxiv_id}",
        f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}",
    ]
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    last_err: Exception | None = None
    async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
        for url in urls:
            try:
                r = await client.get(url, timeout=HTTP_TIMEOUT)
                if r.status_code == 200 and len(r.text) > 1000:
                    parser = _ArxivHTMLTextParser()
                    parser.feed(r.text)
                    text = parser.text()
                    if len(text) > 500:
                        _paper_cache_set(cache_key, (url, text))
                        return url, text
                last_err = RuntimeError(f"{url} returned HTTP {r.status_code}")
            except Exception as e:
                last_err = e
    raise RuntimeError(f"No HTML full text found for {arxiv_id}: {last_err}")


def _section_offsets(text: str) -> list[tuple[int, int, str]]:
    """Return [(offset, level, heading), ...] from markdown-ish text headings."""
    sections: list[tuple[int, int, str]] = []
    for m in re.finditer(r"(?m)^(#{1,4})\s+(.+)$", text):
        heading = re.sub(r"\s+", " ", m.group(2)).strip()
        if heading and len(heading) < 200:
            sections.append((m.start(), len(m.group(1)), heading))
    return sections


def _format_openalex_work(work: dict[str, Any], include_abstract: bool = True) -> str:
    """Render an OpenAlex work record as a compact human-readable block."""
    title = work.get("title") or "(untitled)"
    year = work.get("publication_year") or "n/a"
    cited = work.get("cited_by_count", 0)
    venue = ((work.get("primary_location") or {}).get("source") or {}).get("display_name") or "preprint"
    authors = [a.get("author", {}).get("display_name", "") for a in (work.get("authorships") or []) if a.get("author")]
    authors_str = ", ".join(authors[:5]) + ("…" if len(authors) > 5 else "")
    oa = work.get("open_access") or {}
    pdf_url = oa.get("oa_url") or ""
    ids = work.get("ids") or {}
    doi = ids.get("doi", "")
    openalex_id = (ids.get("openalex") or work.get("id") or "").split("/")[-1]

    arxiv_id = ""
    # arXiv preprints in OpenAlex have DOI 10.48550/arXiv.<id>
    if doi and "arxiv" in doi.lower():
        arxiv_id = doi.lower().split("arxiv.")[-1]

    parts = [f"**{title}**"]
    parts.append(f"Authors: {authors_str or '—'}")
    parts.append(f"Year: {year} | Venue: {venue} | Citations: {cited}")
    id_parts = [f"OpenAlex: {openalex_id}"]
    if doi:
        id_parts.append(f"DOI: {doi.replace('https://doi.org/', '')}")
    if arxiv_id:
        id_parts.append(f"arXiv: {arxiv_id}")
    parts.append(" | ".join(id_parts))
    if pdf_url:
        parts.append(f"PDF: {pdf_url}")
    if include_abstract:
        abstract = _reconstruct_abstract(work.get("abstract_inverted_index"))
        if abstract:
            parts.append(f"\nAbstract:\n{_truncate(abstract)}")
        else:
            parts.append("\nAbstract: (not available)")
    return "\n".join(parts)


def _format_arxiv_entry(entry: dict[str, Any], include_abstract: bool = True) -> str:
    """Render one arXiv Atom entry as a compact block."""
    title = entry.get("title") or "(untitled)"
    authors = entry.get("authors") or []
    authors_str = ", ".join(authors[:5]) + ("..." if len(authors) > 5 else "")
    published = entry.get("published", "n/a")[:10]
    updated = entry.get("updated", "n/a")[:10]
    arxiv_id = entry.get("arxiv_id", "")
    categories = ", ".join(entry.get("categories") or [])
    pdf_url = entry.get("pdf_url") or ""
    abs_url = entry.get("abs_url") or (f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "")

    parts = [f"**{title}**"]
    parts.append(f"Authors: {authors_str or '-'}")
    parts.append(f"Published: {published} | Updated: {updated}")
    id_parts = []
    if arxiv_id:
        id_parts.append(f"arXiv: {arxiv_id}")
    if categories:
        id_parts.append(f"Categories: {categories}")
    if id_parts:
        parts.append(" | ".join(id_parts))
    if abs_url:
        parts.append(f"Abstract URL: {abs_url}")
    if pdf_url:
        parts.append(f"PDF: {pdf_url}")
    if include_abstract:
        abstract = entry.get("summary") or ""
        parts.append(f"\nAbstract:\n{_truncate(abstract)}" if abstract else "\nAbstract: (not available)")
    return "\n".join(parts)


def _parse_arxiv_atom(feed_text: str) -> list[dict[str, Any]]:
    """Parse the arXiv Atom response with stdlib ElementTree."""
    import xml.etree.ElementTree as ET

    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    root = ET.fromstring(feed_text)
    entries: list[dict[str, Any]] = []
    for e in root.findall("atom:entry", ns):

        def text(path: str) -> str:
            node = e.find(path, ns)
            return re.sub(r"\s+", " ", node.text or "").strip() if node is not None else ""

        links = e.findall("atom:link", ns)
        abs_url = ""
        pdf_url = ""
        for link in links:
            href = link.attrib.get("href", "")
            rel = link.attrib.get("rel", "")
            title = link.attrib.get("title", "")
            typ = link.attrib.get("type", "")
            if rel == "alternate":
                abs_url = href
            if title == "pdf" or typ == "application/pdf" or "/pdf/" in href:
                pdf_url = href
        entry_id = text("atom:id")
        arxiv_id = entry_id.rsplit("/", 1)[-1] if entry_id else ""
        entries.append(
            {
                "title": text("atom:title"),
                "summary": text("atom:summary"),
                "published": text("atom:published"),
                "updated": text("atom:updated"),
                "arxiv_id": arxiv_id,
                "abs_url": abs_url,
                "pdf_url": pdf_url,
                "authors": [
                    re.sub(r"\s+", " ", (a.findtext("atom:name", default="", namespaces=ns))).strip()
                    for a in e.findall("atom:author", ns)
                ],
                "categories": [
                    c.attrib.get("term", "") for c in e.findall("atom:category", ns) if c.attrib.get("term")
                ],
            }
        )
    return entries


async def _arxiv_api_search(query: str, max_results: int) -> str:
    """Search arXiv's native Atom API with polite rate limiting."""
    await _arxiv_rate_limit()
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    headers = {"User-Agent": USER_AGENT, "Accept": "application/atom+xml"}
    async with httpx.AsyncClient(headers=headers) as client:
        r = await client.get(ARXIV_BASE, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
    entries = _parse_arxiv_atom(r.text)
    if not entries:
        return f"No arXiv papers found for query: {query!r}."
    return "\n\n---\n\n".join(_format_arxiv_entry(e) for e in entries)


async def _http_get_json(client: httpx.AsyncClient, url: str, params: dict[str, Any] | None = None) -> Any:
    """GET JSON with retry on 429/5xx and exponential backoff.

    Special handling: if OpenAlex returns 429 with "Insufficient budget"
    (premium key exhausted), we drop the api_key param and retry immediately
    on the polite-pool tier instead of burning all retries on a dead key.
    """
    global _key_exhausted
    delay = INITIAL_BACKOFF
    last_err: Exception | None = None
    for attempt in range(NUM_RETRIES + 1):
        try:
            r = await client.get(url, params=params, timeout=HTTP_TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                body_preview = r.text[:300]
                if "Insufficient budget" in body_preview and params and "api_key" in params:
                    if not _key_exhausted:
                        logger.warning("OpenAlex API key budget exhausted — dropping to polite-pool tier")
                        _key_exhausted = True
                    params = {k: v for k, v in params.items() if k != "api_key"}
                    await asyncio.sleep(0.5)
                    continue
                last_err = RuntimeError(f"HTTP 429: {body_preview}")
                if attempt < NUM_RETRIES:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, MAX_BACKOFF)
                    continue
                raise last_err
            if r.status_code in (500, 502, 503, 504):
                last_err = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
                if attempt < NUM_RETRIES:
                    await asyncio.sleep(delay)
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


async def _arxiv_rate_limit() -> None:
    """Throttle arXiv API requests across concurrent workers."""
    global _arxiv_last_request

    # Cross-process/node throttling for cluster evals. This keeps multiple
    # chunk workers from stampeding export.arxiv.org and getting HTTP 429.
    def _locked_sleep() -> None:
        import fcntl

        lock_path = ARXIV_RATE_LIMIT_LOCK
        lock_dir = os.path.dirname(lock_path)
        try:
            os.makedirs(lock_dir, exist_ok=True)
        except Exception:
            lock_path = "/tmp/arxiv_api_rate_limit.lock"
            os.makedirs(os.path.dirname(lock_path), exist_ok=True)

        with open(lock_path, "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            raw = f.read().strip()
            try:
                last = float(raw) if raw else 0.0
            except ValueError:
                last = 0.0
            now = time.time()
            wait = ARXIV_REQUEST_INTERVAL - (now - last)
            if wait > 0:
                time.sleep(wait)
                now = time.time()
            f.seek(0)
            f.truncate()
            f.write(str(now))
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
            fcntl.flock(f, fcntl.LOCK_UN)

    try:
        await asyncio.to_thread(_locked_sleep)
        return
    except Exception as e:
        logger.warning("Shared arXiv rate limiter failed (%s); falling back to process-local limiter", e)

    async with _arxiv_lock:
        now = time.monotonic()
        wait = ARXIV_REQUEST_INTERVAL - (now - _arxiv_last_request)
        if wait > 0:
            await asyncio.sleep(wait)
        _arxiv_last_request = time.monotonic()


# ── Tools ──────────────────────────────────────────────────────────────────


async def arxiv_search(
    query: Annotated[
        str,
        Field(
            description=(
                "Natural-language arXiv query (e.g. 'graphene Hall conductivity' "
                "or 'cat:quant-ph quantum error correction'). Returns compact "
                "metadata/abstracts. Use arxiv-sections/arxiv-read-chunk for paper text."
            ),
        ),
    ],
    max_results: Annotated[int, Field(description="Maximum number of papers to return.")] = 3,
) -> str:
    """Search arXiv papers by keywords/categories using arXiv's native API.

    Returns titles, authors, dates, arXiv IDs, URLs, categories, and a
    truncated abstract. This path intentionally avoids OpenAlex to eliminate
    API-key budget/rate failures during large ablations.
    """
    if max_results < 1:
        return "max_results must be >= 1."
    if max_results > MAX_RESULTS:
        max_results = MAX_RESULTS

    cache_key = _cache_key("arxiv-search", query, max_results)
    if cached := _cache_get(cache_key):
        return cached

    try:
        result = await _arxiv_api_search(query, max_results)
    except Exception as e:
        return f"arXiv search failed: {e}"
    _cache_set(cache_key, result)
    return result


async def arxiv_get(
    paper_id: Annotated[
        str,
        Field(
            description=(
                "Paper identifier. Accepts an arXiv id (e.g. '2103.15348' or "
                "'2103.15348v2'), a DOI ('10.1038/s41586-021-03819-2'), a "
                "PMID ('pmid:34567890'), or an OpenAlex id ('W2741809807')."
            ),
        ),
    ],
) -> str:
    """Fetch full metadata + abstract for a specific paper by ID.

    Returns title, all authors, venue, year, citation count, IDs (OpenAlex /
    DOI / arXiv / PMID), open-access PDF link, and the full reconstructed
    abstract. Backend: OpenAlex.
    """
    norm = _normalize_id(paper_id)
    cache_key = _cache_key("oa-get", norm)
    if cached := _cache_get(cache_key):
        return cached

    params: dict[str, Any] = {"mailto": POLITE_EMAIL}
    if _openalex_api_key and not _key_exhausted:
        params["api_key"] = _openalex_api_key
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    url = f"{OPENALEX_BASE}/works/{norm}"
    async with httpx.AsyncClient(headers=headers) as client:
        try:
            work = await _http_get_json(client, url, params=params)
        except Exception as e:
            # Fallback: bare arXiv API (with rate limit) for arxiv ids.
            if arxiv_id := _extract_arxiv_id(paper_id):
                try:
                    return await _arxiv_api_get(arxiv_id)
                except Exception as e2:
                    return f"Paper lookup failed (OpenAlex: {e}; arXiv: {e2})."
            return f"Paper lookup failed (OpenAlex): {e}"

    if not work or "id" not in work:
        return f"Paper {paper_id!r} not found."
    rendered = _format_openalex_work(work, include_abstract=True)
    _cache_set(cache_key, rendered)
    return rendered


async def arxiv_sections(
    paper_id: Annotated[
        str,
        Field(description="arXiv id or URL (e.g. '2103.15348', 'arXiv:2103.15348', or an arxiv.org/abs URL)."),
    ],
    max_sections: Annotated[int, Field(description="Maximum section headings to return (1-80).")] = 40,
) -> str:
    """List section headings for a paper without returning the full paper.

    Prefer this after `arxiv-search`/`arxiv-get` when you need paper content:
    first inspect the section map, then call `arxiv-read-chunk` at the relevant
    offset or with a query.
    """
    max_sections = max(1, min(int(max_sections or 40), 80))
    cache_key = _cache_key("paper-sections", paper_id, max_sections)
    if cached := _cache_get(cache_key):
        return cached
    try:
        source, text = await _fetch_paper_text(paper_id)
    except Exception as e:
        return f"arXiv section listing failed: {e}"
    sections = _section_offsets(text)
    if not sections:
        rendered = (
            f"Full text loaded from {source}, but no section headings were parsed.\n"
            f"total_chars={len(text)}. Use arxiv-read-chunk with offset=0."
        )
    else:
        lines = [
            f"Source: {source}",
            f"total_chars={len(text)}; use arxiv-read-chunk(paper_id, offset=...) for bounded text.",
            "",
        ]
        for off, level, heading in sections[:max_sections]:
            lines.append(f"- offset={off} | L{level} | {heading}")
        if len(sections) > max_sections:
            lines.append(f"... {len(sections) - max_sections} more sections omitted")
        rendered = "\n".join(lines)
    _cache_set(cache_key, rendered)
    return rendered


async def arxiv_read_chunk(
    paper_id: Annotated[
        str,
        Field(description="arXiv id or URL (e.g. '2103.15348', 'arXiv:2103.15348', or an arxiv.org/abs URL)."),
    ],
    offset: Annotated[
        int, Field(description="Character offset into the extracted paper text. Use arxiv-sections offsets.")
    ] = 0,
    max_chars: Annotated[int, Field(description="Maximum characters to return (500-6000).")] = PAPER_CHUNK_LIMIT,
    query: Annotated[
        str,
        Field(
            description="Optional term/phrase to locate; if provided, returns a chunk centered near the first match."
        ),
    ] = "",
) -> str:
    """Read a bounded chunk of arXiv full text.

    This avoids dumping whole papers into context. It uses native arXiv HTML
    first and falls back to ar5iv, similar to popular arXiv MCPs.
    """
    max_chars = max(500, min(int(max_chars or PAPER_CHUNK_LIMIT), 6000))
    offset = max(0, int(offset or 0))
    cache_key = _cache_key("paper-chunk", paper_id, offset, max_chars, query)
    if cached := _cache_get(cache_key):
        return cached
    try:
        source, text = await _fetch_paper_text(paper_id)
    except Exception as e:
        return f"arXiv chunk read failed: {e}"

    note = ""
    if query and query.strip():
        q = query.strip()
        idx = text.lower().find(q.lower())
        if idx >= 0:
            offset = max(0, idx - max_chars // 2)
            note = f"Chunk centered around query {q!r}.\n"
        else:
            note = f"Query {q!r} not found exactly; using offset={offset}.\n"
    end = min(len(text), offset + max_chars)
    chunk = text[offset:end].strip()
    next_offset = end if end < len(text) else None
    rendered = (
        f"Source: {source}\noffset={offset}; next_offset={next_offset}; total_chars={len(text)}\n{note}\n{chunk}"
    )
    _cache_set(cache_key, rendered)
    return rendered


async def _arxiv_api_get(arxiv_id: str) -> str:
    """Fallback: hit arXiv's own API for a single paper. Rate-limited."""
    await _arxiv_rate_limit()
    params = {"id_list": arxiv_id, "max_results": 1}
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(headers=headers) as client:
        r = await client.get(ARXIV_BASE, params=params, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        text = r.text

    if "<entry>" not in text:
        return f"Paper {arxiv_id!r} not found."

    # Tiny Atom parser — extract title and summary without depending on lxml.
    def _grab(tag: str) -> str:
        m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", text, re.DOTALL)
        return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""

    title = _grab("title") or "(no title)"
    # The first <title> in an arXiv feed is the feed title; take the second.
    titles = re.findall(r"<entry>.*?<title[^>]*>(.*?)</title>", text, re.DOTALL)
    if titles:
        title = re.sub(r"\s+", " ", titles[0]).strip()
    summary = ""
    sm = re.search(r"<summary[^>]*>(.*?)</summary>", text, re.DOTALL)
    if sm:
        summary = re.sub(r"\s+", " ", sm.group(1)).strip()
    pub = re.search(r"<published>(.*?)</published>", text)
    return (
        f"**{title}**\n"
        f"arXiv: {arxiv_id}\n"
        f"Published: {pub.group(1)[:10] if pub else 'n/a'}\n\n"
        f"Abstract:\n{_truncate(summary)}"
    )


# ── Tool provider ──────────────────────────────────────────────────────────


class ArxivSearchTool(Tool):
    """Direct arXiv search/retrieval tool."""

    def __init__(self) -> None:
        self._config: dict[str, Any] = {
            "max_results": 3,
            "max_sections": 40,
            "max_chars": PAPER_CHUNK_LIMIT,
        }

    def default_config(self) -> dict[str, Any]:
        return dict(self._config)

    def configure(self, overrides: dict[str, Any] | None = None, context: dict[str, Any] | None = None) -> None:
        if not overrides:
            return

        allowed = {"max_results", "max_sections", "max_chars"}
        unknown = set(overrides) - allowed
        if unknown:
            raise ValueError(f"Unsupported ArxivSearchTool override(s): {sorted(unknown)}")

        if "max_results" in overrides:
            self._config["max_results"] = max(1, min(int(overrides["max_results"]), MAX_RESULTS))
        if "max_sections" in overrides:
            self._config["max_sections"] = max(1, min(int(overrides["max_sections"]), 80))
        if "max_chars" in overrides:
            self._config["max_chars"] = max(500, min(int(overrides["max_chars"]), PAPER_CHUNK_LIMIT))

    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "arxiv-search",
                "description": "Search arXiv papers by keywords/categories and return compact metadata and abstracts.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "Natural-language arXiv query."}},
                    "required": ["query"],
                },
            },
            {
                "name": "arxiv-get",
                "description": "Fetch metadata and abstract for a paper by arXiv id, DOI, PMID, or OpenAlex id.",
                "input_schema": {
                    "type": "object",
                    "properties": {"paper_id": {"type": "string", "description": "Paper identifier."}},
                    "required": ["paper_id"],
                },
            },
            {
                "name": "arxiv-sections",
                "description": "List parsed section headings for an arXiv paper.",
                "input_schema": {
                    "type": "object",
                    "properties": {"paper_id": {"type": "string", "description": "arXiv id or URL."}},
                    "required": ["paper_id"],
                },
            },
            {
                "name": "arxiv-read-chunk",
                "description": "Read a bounded chunk of arXiv full text by offset or query.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "paper_id": {"type": "string", "description": "arXiv id or URL."},
                        "offset": {"type": "integer", "description": "Character offset into the extracted text."},
                        "query": {"type": "string", "description": "Optional phrase to center the chunk around."},
                    },
                    "required": ["paper_id"],
                },
            },
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any], extra_args: dict[str, Any] | None = None):
        arguments = dict(arguments or {})
        if tool_name == "arxiv-search":
            arguments.setdefault("max_results", self._config["max_results"])
            return await arxiv_search(**arguments)
        if tool_name == "arxiv-get":
            return await arxiv_get(**arguments)
        if tool_name == "arxiv-sections":
            arguments.setdefault("max_sections", self._config["max_sections"])
            return await arxiv_sections(**arguments)
        if tool_name == "arxiv-read-chunk":
            arguments.setdefault("offset", 0)
            arguments.setdefault("max_chars", self._config["max_chars"])
            arguments.setdefault("query", "")
            return await arxiv_read_chunk(**arguments)
        return f"Error: unknown tool '{tool_name}'"
