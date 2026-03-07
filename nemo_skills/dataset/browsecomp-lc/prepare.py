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

# Prompt structure adapted from render_prompt() in
# https://huggingface.co/datasets/openai/BrowseCompLongContext
# Decryption adapted from browsecomp_eval.py:
# https://github.com/openai/simple-evals/blob/main/browsecomp_eval.py

import argparse
import base64
import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import html2text
import requests
import tiktoken
from datasets import load_dataset
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

MAX_WORKERS = 16  # parallel URL fetches
MAX_CONTENT_BYTES = 5 * 1024 * 1024  # 5 MB streaming limit


# ---------------------------------------------------------------------------
# Decryption (from browsecomp_eval.py)
# ---------------------------------------------------------------------------


def _derive_key(password: str, length: int) -> bytes:
    key = hashlib.sha256(password.encode()).digest()
    return key * (length // len(key)) + key[: length % len(key)]


def _decrypt(ciphertext_b64: str, password: str) -> str:
    encrypted = base64.b64decode(ciphertext_b64)
    key = _derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode()


# ---------------------------------------------------------------------------
# HTTP session with automatic retries (per-thread)
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=0)  # no retries — all error cases handled in _fetch_and_cache
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    # Use a browser UA — many sites 403 bot-identified strings.
    session.headers["User-Agent"] = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    return session


def _get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        _thread_local.session = _make_session()
    return _thread_local.session


# ---------------------------------------------------------------------------
# URL fetching helpers
# ---------------------------------------------------------------------------

_SENTINEL_MISSING = object()  # distinguishes "not in cache" from "cached as None"


def _fetch_url_text(url: str, session: requests.Session) -> str:
    """Fetch a URL and convert HTML → plain text (matching _fetch_url in render_prompt).
    Streams response and aborts after MAX_CONTENT_BYTES."""
    resp = session.get(url, timeout=(5, 15), stream=True)
    resp.raise_for_status()

    chunks: list[bytes] = []
    size = 0
    for chunk in resp.iter_content(chunk_size=65536, decode_unicode=False):
        size += len(chunk)
        chunks.append(chunk)
        if size > MAX_CONTENT_BYTES:
            break
    raw = b"".join(chunks).decode("utf-8", errors="replace")

    h = html2text.HTML2Text()
    h.ignore_links = True
    h.ignore_images = True
    h.body_width = 0
    text = h.handle(raw)

    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Treat near-empty responses as permanent failures — JS-rendered pages
    # return 200 but the HTML skeleton strips to almost nothing without JS.
    # Raise a non-RequestException so _fetch_and_cache negatively caches it.
    if len(text) < 200:
        raise ValueError(f"Response body too short ({len(text)} chars) — likely JS-rendered")
    return text


# ---------------------------------------------------------------------------
# Per-URL disk cache (one JSON file per URL, keyed by SHA-256 of URL).
# Negative caching: content=null means the URL permanently failed.
# ---------------------------------------------------------------------------


def _cache_path(url: str, cache_dir: Path) -> Path:
    return cache_dir / (hashlib.sha256(url.encode()).hexdigest() + ".json")


def _load_cached(url: str, cache_dir: Path) -> str | None | object:
    """Return content string, None (negatively cached), or _SENTINEL_MISSING."""
    p = _cache_path(url, cache_dir)
    if p.exists():
        return json.loads(p.read_text())["content"]  # may be None
    return _SENTINEL_MISSING


def _save_cached(url: str, content: str | None, cache_dir: Path) -> None:
    p = _cache_path(url, cache_dir)
    p.write_text(json.dumps({"url": url, "content": content}))


# HTTP status codes that indicate a permanent failure — safe to negatively cache.
# Transient failures (timeout, connection error, 429, 5xx) are NOT cached so
# they are retried on the next run.
_PERMANENT_FAILURE_CODES = frozenset({400, 401, 403, 404, 405, 410, 451})

# Wayback Machine CDX API endpoint (JSON) — used to find the closest archived
# snapshot when the live URL fails permanently.
_WAYBACK_CDX = "https://web.archive.org/web/20231201000000*/{url}"


def _fetch_with_wayback(url: str, session: requests.Session) -> str | None:
    """Try live URL, then Wayback Machine snapshot on permanent failure."""
    try:
        return _fetch_url_text(url, session)
    except requests.exceptions.HTTPError as exc:
        if exc.response is None or exc.response.status_code not in _PERMANENT_FAILURE_CODES:
            raise  # transient — let caller handle
        # Permanent failure: try Wayback Machine
        wayback = _WAYBACK_CDX.format(url=url)
        try:
            return _fetch_url_text(wayback, session)
        except requests.exceptions.RequestException:
            return None
    # Non-HTTP errors (timeout, DNS, etc.) propagate to caller as transient.


def _fetch_and_cache(url: str, cache_dir: Path) -> str | None:
    """Fetch url (skipping if cached), persist result. Returns content or None.

    Only negatively caches permanent failures; transient failures leave the
    cache empty so the URL is retried on the next run.
    """
    cached = _load_cached(url, cache_dir)
    if cached is not _SENTINEL_MISSING:
        return cached  # type: ignore[return-value]

    session = _get_session()
    try:
        content = _fetch_with_wayback(url, session)
    except requests.exceptions.RequestException:
        # Transient failure (timeout, connection error, 429, 5xx, etc.) —
        # do NOT write to cache; will be retried next run.
        return None
    except Exception:
        # Malformed URL or other permanent parse error — negatively cache so
        # it is not retried (e.g. urllib3.exceptions.LocationParseError).
        content = None

    # content is None only for permanent failures (already tried Wayback).
    _save_cached(url, content, cache_dir)
    return content


def _get_page_text(url: str, cache_dir: Path) -> str | None:
    """Return page text from cache only (no network I/O expected after pre-fetch)."""
    cached = _load_cached(url, cache_dir)
    if cached is _SENTINEL_MISSING:
        return None  # shouldn't happen after pre-fetch
    return cached  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Token counting and page fitting
# ---------------------------------------------------------------------------


def _count_tokens(text: str, enc: tiktoken.Encoding | None) -> int:
    assert enc is not None, "enc required when token_budget is set"
    return len(enc.encode(text))


def _fit_pages(pages: list[str], budget: int | None, enc: tiktoken.Encoding | None) -> tuple[int, int | None]:
    """Greedily pack pages into budget (stop at first page that doesn't fit).
    If budget is None, all pages fit unconditionally."""
    if budget is None:
        return len(pages), None
    fitted = 0
    for page in pages:
        n = _count_tokens(page, enc)
        if n <= budget:
            budget -= n
            fitted += 1
        else:
            break
    return fitted, budget


# ---------------------------------------------------------------------------
# Prompt construction (mirrors render_prompt from dataset README)
# ---------------------------------------------------------------------------

_INITIAL_TEMPLATE = (
    "Given a list of websites, answer the following question: {problem}\n\n"
    "Your final answer should be a concise sentence, in the following format:\n"
    "Final Answer: put your answer here.\n\n"
    "It's critical your answer is concise and following the format strictly.\n"
)

_FINAL_TEMPLATE = (
    "\nNow answer the original question, recall the question is: {problem}\n\n"
    "VERY IMPORTANT: Do not use any web search tools or browser tools to answer "
    "the question, you may only use the provided documents to answer the question."
)


def _shuffle_pages(pages: list[str], seed: str) -> list[str]:
    """Deterministic shuffle via hash sort — stable across Python versions."""
    return sorted(pages, key=lambda p: hashlib.sha256((seed + p).encode()).digest())


def _build_prompt(
    problem: str,
    required_pages: list[str],
    additional_pages: list[str],
    token_budget: int | None,
    enc: tiktoken.Encoding | None,
) -> str | None:
    initial = _INITIAL_TEMPLATE.format(problem=problem)
    final = _FINAL_TEMPLATE.format(problem=problem)

    budget = (
        None
        if token_budget is None
        else token_budget - _count_tokens(initial, enc) - _count_tokens(final, enc)
    )

    num_req, budget = _fit_pages(required_pages, budget, enc)
    if num_req < len(required_pages):
        return None  # required pages exceed budget; example is skipped

    num_add, _ = _fit_pages(additional_pages, budget, enc)

    pages = _shuffle_pages(
        required_pages[:num_req] + additional_pages[:num_add],
        seed=problem,
    )

    return "\n".join([initial, *pages, final])


# ---------------------------------------------------------------------------
# Decrypt dataset entries
# ---------------------------------------------------------------------------


def _decrypt_entry(entry: dict) -> dict:
    canary = entry["canary"]
    problem = _decrypt(entry["problem"], canary)
    answer = _decrypt(entry["answer"], canary)
    url_pairs: list[list[str]] = json.loads(_decrypt(entry["urls"], canary))

    required_urls = [u for u, label in url_pairs if label == "required"]
    additional_urls = [u for u, label in url_pairs if label == "additional"]
    unknown = {label for _, label in url_pairs if label not in ("required", "additional")}
    assert not unknown, f"Unexpected URL labels: {unknown}"

    return {
        "problem": problem,
        "answer": answer,
        "required_urls": required_urls,
        "additional_urls": additional_urls,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BrowseCompLongContext dataset.")
    parser.add_argument(
        "--token_budget",
        type=str,
        default=None,
        help="Maximum tokens per prompt, e.g. 128k, 256k, 1m. If unset, all pages are included.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Limit number of examples processed (for testing).",
    )
    parser.add_argument(
        "--max_urls",
        type=int,
        default=None,
        help="Limit number of URLs fetched per example (for testing).",
    )
    parser.add_argument(
        "--force_refetch",
        action="store_true",
        help="Ignore cache (including negative cache) and re-fetch all URLs.",
    )
    parser.add_argument(
        "--cached_only",
        action="store_true",
        help="Skip examples where any URL is not in cache (no network I/O). "
             "Useful for correctness testing after cache is warm.",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Decrypt all examples, check cache (no network), and print a summary "
             "of which examples have uncached URLs. Does not write output.",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).absolute().parent
    cache_dir = data_dir / "_cache"
    cache_dir.mkdir(exist_ok=True)

    if args.token_budget is not None:
        _s = args.token_budget.strip().lower()
        if _s.endswith("m"):
            token_budget = int(float(_s[:-1]) * 1024 * 1024)
        elif _s.endswith("k"):
            token_budget = int(float(_s[:-1]) * 1024)
        else:
            token_budget = int(_s)
        # Normalise label: e.g. 131072 → "128k", 262144 → "256k"
        _label = args.token_budget.strip().lower().rstrip("0")
        output_file = data_dir / f"{_label}.jsonl"
    else:
        token_budget = None
        output_file = data_dir / "test.jsonl"

    enc = tiktoken.get_encoding("o200k_base") if token_budget is not None else None

    dataset = load_dataset("openai/BrowseCompLongContext")["train"]
    if args.max_examples is not None:
        dataset = dataset.select(range(min(args.max_examples, len(dataset))))
    raw_entries = list(dataset)

    # Phase 1: decrypt all examples
    print(f"Decrypting {len(raw_entries)} examples…")
    examples = [_decrypt_entry(e) for e in tqdm(raw_entries, desc="Decrypting")]

    # Apply --max_urls truncation
    if args.max_urls is not None:
        for ex in examples:
            ex["required_urls"] = ex["required_urls"][: args.max_urls]
            ex["additional_urls"] = ex["additional_urls"][: max(0, args.max_urls - len(ex["required_urls"]))]

    # --scan mode: report cache status, then exit
    if args.scan:
        fully_cached = 0
        partially_missing: list[tuple[str, int]] = []
        for ex in examples:
            all_urls = ex["required_urls"] + ex["additional_urls"]
            missing = [u for u in all_urls if _load_cached(u, cache_dir) is _SENTINEL_MISSING]
            if not missing:
                fully_cached += 1
            else:
                partially_missing.append((ex["problem"][:80], len(missing)))
        print(f"\nScan results ({len(examples)} examples):")
        print(f"  Fully cached:    {fully_cached}")
        print(f"  Has uncached:    {len(partially_missing)}")
        if partially_missing:
            print("\nExamples with uncached URLs (problem prefix → uncached count):")
            for prob, n in partially_missing:
                print(f"  [{n:3d} missing] {prob!r}")
        return

    # Phase 2+3 (interleaved): submit all URL fetches, write each example as soon as
    # its own URLs complete — no waiting for the entire dataset's URLs to finish first.

    # Load already-written problems to support resuming.
    done_problems: set[str] = set()
    if output_file.exists():
        valid_lines = []
        with open(output_file, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    done_problems.add(json.loads(line)["problem"])
                    valid_lines.append(line.rstrip("\n"))
                except (json.JSONDecodeError, KeyError):
                    pass
        with open(output_file, "wt", encoding="utf-8") as f:
            for line in valid_lines:
                f.write(line + "\n")
        if done_problems:
            print(f"Resuming — {len(done_problems)} examples already written.")

    written = len(done_problems)
    skipped = 0

    # Build per-URL future map (empty when --cached_only).
    url_futures: dict[str, "Future[str | None]"] = {}
    executor = None

    if args.cached_only:
        print("\n--cached_only: skipping network fetch phase.")
    else:
        # Collect uncached URLs in dataset order so early examples' URLs are
        # submitted to the executor first and complete soonest, preventing a
        # later example's URL from blocking the sequential example loop.
        seen_urls: set[str] = set()
        ordered_uncached: list[str] = []
        for ex in examples:
            for url in ex["required_urls"] + ex["additional_urls"]:
                if url not in seen_urls:
                    seen_urls.add(url)
                    if args.force_refetch:
                        _cache_path(url, cache_dir).unlink(missing_ok=True)
                        ordered_uncached.append(url)
                    elif _load_cached(url, cache_dir) is _SENTINEL_MISSING:
                        ordered_uncached.append(url)
        all_urls = seen_urls
        uncached_urls = ordered_uncached

        if uncached_urls:
            n_cached = len(all_urls) - len(uncached_urls)
            print(
                f"\n{len(all_urls)} unique URLs — {n_cached} already cached, "
                f"fetching {len(uncached_urls)} (workers={MAX_WORKERS})…"
            )
            url_bar = tqdm(total=len(uncached_urls), desc="URLs fetched", leave=True)

            def _on_url_done(_f: "Future") -> None:
                url_bar.update(1)

            executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
            for u in uncached_urls:
                fut = executor.submit(_fetch_and_cache, u, cache_dir)
                fut.add_done_callback(_on_url_done)
                url_futures[u] = fut
        else:
            print("\nAll URLs already cached — skipping network phase.")

    n_todo = sum(1 for ex in examples if ex["problem"] not in done_problems)
    example_bar = tqdm(total=n_todo, desc="Examples processed", leave=True)
    example_bar.set_postfix(written=written, skipped=skipped)

    def _update_bar() -> None:
        example_bar.set_postfix(written=written, skipped=skipped)
        example_bar.update(1)

    try:
        with open(output_file, "at", encoding="utf-8") as fout:
            for ex in examples:
                if ex["problem"] in done_problems:
                    continue

                required_urls = ex["required_urls"]
                additional_urls = ex["additional_urls"]

                # --cached_only: skip examples with any uncached URL
                if args.cached_only:
                    if any(_load_cached(u, cache_dir) is _SENTINEL_MISSING for u in required_urls + additional_urls):
                        skipped += 1
                        _update_bar()
                        continue

                # Wait for this example's URL futures (already done for cached URLs)
                for url in required_urls + additional_urls:
                    if url in url_futures:
                        url_futures[url].result()  # blocks until this URL is fetched

                required_pages_raw = [_get_page_text(u, cache_dir) for u in required_urls]
                additional_pages_raw = [_get_page_text(u, cache_dir) for u in additional_urls]

                unavailable_urls = (
                    [u for u, p in zip(required_urls, required_pages_raw) if p is None]
                    + [u for u, p in zip(additional_urls, additional_pages_raw) if p is None]
                )

                missing_required = [u for u, p in zip(required_urls, required_pages_raw) if p is None]
                if missing_required:
                    tqdm.write(f"[skip] required URL(s) unavailable: {ex['problem'][:80]!r}")
                    skipped += 1
                    _update_bar()
                    continue

                required_pages = [p for p in required_pages_raw if p is not None]
                additional_pages = [p for p in additional_pages_raw if p is not None]

                prompt = _build_prompt(
                    ex["problem"], required_pages, additional_pages, token_budget, enc
                )
                if prompt is None:
                    tqdm.write(
                        f"[skip] required pages exceed token_budget={token_budget}: {ex['problem'][:80]!r}"
                    )
                    skipped += 1
                    _update_bar()
                    continue

                record = {
                    "messages": [{"role": "user", "content": prompt}],
                    "expected_answer": ex["answer"],
                    "problem": ex["problem"],
                }
                if unavailable_urls:
                    record["unavailable_urls"] = unavailable_urls
                json.dump(record, fout)
                fout.write("\n")
                fout.flush()
                written += 1
                _update_bar()
    finally:
        example_bar.close()
        if executor is not None:
            executor.shutdown(wait=True)  # drain remaining fetches into cache

    print(f"\nDone — written: {written}, skipped: {skipped}")
    n_cache = sum(1 for _ in cache_dir.glob("*.json"))
    print(f"URL cache: {n_cache} entries in {cache_dir}")


if __name__ == "__main__":
    main()
