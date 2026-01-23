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


import importlib.util
import os
import random
import time
from typing import Optional

import html2text
import requests
from bs4 import BeautifulSoup

ERROR_TEMPLATES = [
    "503 Server Error: Service Unavailable for url: {url}",
    "429 Client Error: Too Many Requests for url: {url}",
    "403 Client Error: Forbidden for url: {url}",
    (
        "HTTPSConnectionPool(host='{host}', port=443): Max retries exceeded with url: {path} "
        "(Caused by ConnectTimeoutError(<urllib3.connection.HTTPSConnection object at 0x{id1:x}>, "
        "'Connection to {host} timed out. (connect timeout=5)'))"
    ),
    "HTTPSConnectionPool(host='{host}', port=443): Read timed out. (read timeout=5)",
    (
        "Max retries exceeded with url: {path} "
        "(Caused by NewConnectionError('<urllib3.connection.HTTPSConnection object at 0x{id2:x}>: "
        "Failed to establish a new connection: [Errno -2] Name or service not known'))"
    ),
]


class WebSearchBackendUnavailable(RuntimeError):
    """Raised when neither ddgs nor serpapi is available to perform web search."""


class WebSearchAPI:
    def __init__(self):
        self._api_description = "This tool belongs to the Web Search API category. It provides functions to search the web and browse search results."
        self.show_snippet = True
        self._warned_no_serp_key = False
        self._validated_backends = False

    def _load_scenario(self, initial_config: dict, long_context: bool = False):
        # We don't care about the long_context parameter here
        # It's there to match the signature of functions in the multi-turn evaluation code
        self.show_snippet = initial_config["show_snippet"]
        # Validate once per instance so BFCL fails fast before generation starts.
        self._validate_backends_available()

    @staticmethod
    def _get_serp_api_key() -> Optional[str]:
        """
        Returns SERP API key if configured.
        """
        return os.environ.get("SERPAPI_API_KEY")

    @staticmethod
    def _has_module(module_name: str) -> bool:
        # find_spec avoids importing the module (important for optional deps / older containers)
        return importlib.util.find_spec(module_name) is not None

    def _validate_backends_available(self):
        """
        Fail fast if web search cannot possibly work in this environment.

        Rules:
        - If SERPAPI_API_KEY is not set: we require ddgs (native DuckDuckGo) to be importable.
        - If SERPAPI_API_KEY is set: we require at least one usable backend:
            - serpapi (preferred), OR
            - ddgs (fallback if serpapi missing)
        """
        if self._validated_backends:
            return
        self._validated_backends = True

        has_ddgs = self._has_module("ddgs")
        has_serpapi = self._has_module("serpapi")
        serp_key = self._get_serp_api_key()

        if serp_key:
            if not has_serpapi and not has_ddgs:
                raise WebSearchBackendUnavailable(
                    "SERPAPI_API_KEY is set, but neither serpapi nor ddgs is installed. "
                    "Install serpapi to use SerpApi's DuckDuckGo engine or install ddgs to fall back to native DuckDuckGo."
                )
        else:
            if not has_ddgs:
                raise WebSearchBackendUnavailable(
                    "SERPAPI_API_KEY is not set and ddgs is not installed, so web search cannot run. "
                    "Install ddgs to use native DuckDuckGo search, or set SERPAPI_API_KEY and install serpapi."
                )

    def _warn_no_serp_api_key_once(self):
        if self._warned_no_serp_key:
            return
        self._warned_no_serp_key = True
        print(
            (
                "*"
                * 100
                + "\n⚠️  [WebSearchAPI] SERPAPI_API_KEY is not set. Falling back to native DuckDuckGo (ddgs). "
                "This may be rate-limited or blocked under high parallelism. "
                "If you see repeated blocks/rate limits, set SERPAPI_API_KEY to use SerpApi's DuckDuckGo engine."
                + "\n"
                + "*" * 100
            )
        )

    def _format_results(self, results: list[dict]) -> list[dict]:
        """Normalize results into the expected schema for the eval."""
        formatted = []
        for r in results:
            if self.show_snippet:
                formatted.append({"title": r.get("title", ""), "href": r.get("href", ""), "body": r.get("body", "")})
            else:
                formatted.append({"title": r.get("title", ""), "href": r.get("href", "")})
        return formatted

    def _search_with_serpapi_duckduckgo(
        self, *, keywords: str, max_results: int, region: str, api_key: str
    ) -> list[dict]:
        """
        Use SerpApi's DuckDuckGo engine (not ddgs) when SERPAPI_API_KEY is available.
        """
        try:
            # Dynamic import: allow running in older containers without serpapi installed,
            # as long as ddgs is available and SERPAPI_API_KEY is not required.
            from serpapi.google_search import GoogleSearch  # type: ignore
        except Exception as e:
            # Let the caller decide whether to fall back to ddgs or fail hard.
            raise ModuleNotFoundError("serpapi is not installed") from e

        backoff = 2  # initial back-off in seconds (matches the reference implementation)

        # SerpApi expects engine='duckduckgo'
        params = {
            "engine": "duckduckgo",
            "q": keywords,
            "kl": region,
            "api_key": api_key,
        }

        # Infinite retry loop with exponential backoff on 429s (matches reference behavior)
        while True:
            try:
                search_results = GoogleSearch(params).get_dict()
            except Exception as e:
                # If the underlying HTTP call raised a 429 we retry, otherwise propagate as an error
                if "429" in str(e):
                    wait_time = backoff + random.uniform(0, backoff)
                    error_block = (
                        "*"
                        * 100
                        + "\n❗️❗️ [WebSearchAPI] Received 429 from SerpAPI. The number of requests sent using this API key "
                        "exceeds the hourly throughput limit OR your account has run out of searches. "
                        f"Retrying in {wait_time:.1f} seconds…" + "*" * 100
                    )
                    print(error_block)
                    time.sleep(wait_time)
                    backoff = min(backoff * 2, 120)  # cap the back-off
                    continue
                error_block = (
                    "*" * 100
                    + f"\n❗️❗️ [WebSearchAPI] Error from SerpAPI: {str(e)}. This is not a rate-limit error, so it will not be retried."
                    + "*" * 100
                )
                print(error_block)
                return {"error": str(e)}

            # SerpAPI sometimes returns the error in the payload instead of raising
            if "error" in search_results and "429" in str(search_results["error"]):
                wait_time = backoff + random.uniform(0, backoff)
                error_block = (
                    "*"
                    * 100
                    + "\n❗️❗️ [WebSearchAPI] Received 429 from SerpAPI. The number of requests sent using this API key "
                    "exceeds the hourly throughput limit OR your account has run out of searches. "
                    f"Retrying in {wait_time:.1f} seconds…" + "*" * 100
                )
                print(error_block)
                time.sleep(wait_time)
                backoff = min(backoff * 2, 120)
                continue

            break  # Success – no rate-limit error detected

        organic = search_results.get("organic_results") or []
        if not organic:
            # Keep behavior aligned with reference (no organic_results => error)
            return {"error": "Failed to retrieve the search results from server. Please try again later."}

        out: list[dict] = []
        for item in organic[:max_results]:
            out.append(
                {
                    "title": item.get("title", ""),
                    "href": item.get("link", ""),
                    "body": item.get("snippet", ""),
                }
            )
        return out

    def _search_with_ddgs(self, *, keywords: str, max_results: int, region: str) -> list[dict]:
        try:
            # Dynamic import: allow running in older containers without ddgs installed,
            # as long as serpapi is available when SERPAPI_API_KEY is set.
            from ddgs import DDGS  # type: ignore
        except Exception:
            raise WebSearchBackendUnavailable(
                "Neither ddgs nor serpapi is available for web search. "
                "Install ddgs to use native DuckDuckGo search, or set SERPAPI_API_KEY and install serpapi to use SerpApi."
            )

        # DDGS.text() may return an iterator; normalize to list.
        search_results = DDGS(timeout=60).text(
            query=keywords, region=region, max_results=max_results, backend="duckduckgo"
        )
        results_list = list(search_results) if search_results else []
        out = []
        for result in results_list[:max_results]:
            out.append(
                {
                    "title": result.get("title", ""),
                    "href": result.get("href", ""),
                    "body": result.get("body", ""),
                }
            )
        return out

    def search_engine_query(
        self,
        keywords: str,
        max_results: Optional[int] = 10,
        region: Optional[str] = "wt-wt",
    ) -> list:
        """
        This function queries the search engine for the provided keywords and region.

        Args:
            keywords (str): The keywords to search for.
            max_results (int, optional): The maximum number of search results to return. Defaults to 10.
            region (str, optional): The region to search in. Defaults to "wt-wt". Possible values include:
                - xa-ar for Arabia
                - xa-en for Arabia (en)
                - ar-es for Argentina
                - au-en for Australia
                - at-de for Austria
                - be-fr for Belgium (fr)
                - be-nl for Belgium (nl)
                - br-pt for Brazil
                - bg-bg for Bulgaria
                - ca-en for Canada
                - ca-fr for Canada (fr)
                - ct-ca for Catalan
                - cl-es for Chile
                - cn-zh for China
                - co-es for Colombia
                - hr-hr for Croatia
                - cz-cs for Czech Republic
                - dk-da for Denmark
                - ee-et for Estonia
                - fi-fi for Finland
                - fr-fr for France
                - de-de for Germany
                - gr-el for Greece
                - hk-tzh for Hong Kong
                - hu-hu for Hungary
                - in-en for India
                - id-id for Indonesia
                - id-en for Indonesia (en)
                - ie-en for Ireland
                - il-he for Israel
                - it-it for Italy
                - jp-jp for Japan
                - kr-kr for Korea
                - lv-lv for Latvia
                - lt-lt for Lithuania
                - xl-es for Latin America
                - my-ms for Malaysia
                - my-en for Malaysia (en)
                - mx-es for Mexico
                - nl-nl for Netherlands
                - nz-en for New Zealand
                - no-no for Norway
                - pe-es for Peru
                - ph-en for Philippines
                - ph-tl for Philippines (tl)
                - pl-pl for Poland
                - pt-pt for Portugal
                - ro-ro for Romania
                - ru-ru for Russia
                - sg-en for Singapore
                - sk-sk for Slovak Republic
                - sl-sl for Slovenia
                - za-en for South Africa
                - es-es for Spain
                - se-sv for Sweden
                - ch-de for Switzerland (de)
                - ch-fr for Switzerland (fr)
                - ch-it for Switzerland (it)
                - tw-tzh for Taiwan
                - th-th for Thailand
                - tr-tr for Turkey
                - ua-uk for Ukraine
                - uk-en for United Kingdom
                - us-en for United States
                - ue-es for United States (es)
                - ve-es for Venezuela
                - vn-vi for Vietnam
                - wt-wt for No region

        Returns:
            list: A list of search result dictionaries, each containing information such as:
            - 'title' (str): The title of the search result.
            - 'href' (str): The URL of the search result.
            - 'body' (str): A brief description or snippet from the search result.
        """
        backoff = 10  # initial back-off in seconds

        # Prefer SerpApi if configured (more reliable under parallelism / rate limits).
        serp_api_key = self._get_serp_api_key()
        if serp_api_key:
            try:
                serp_results = self._search_with_serpapi_duckduckgo(
                    keywords=keywords,
                    max_results=max_results or 10,
                    region=region or "wt-wt",
                    api_key=serp_api_key,
                )
                # _search_with_serpapi_duckduckgo may return {"error": "..."} for non-retryable failures
                if isinstance(serp_results, dict) and "error" in serp_results:
                    return serp_results
                return self._format_results(serp_results)
            except ModuleNotFoundError:
                # Optional requirement: if serpapi isn't installed in an older container, warn and fall back to ddgs.
                # If ddgs is also missing, _search_with_ddgs will raise RuntimeError (hard fail).
                print(
                    (
                        "*" * 100 + "\n⚠️  [WebSearchAPI] SERPAPI_API_KEY is set but serpapi is not installed. "
                        "Falling back to native DuckDuckGo (ddgs)." + "\n" + "*" * 100
                    )
                )
        else:
            self._warn_no_serp_api_key_once()

        # Infinite retry loop with exponential backoff
        while True:
            try:
                wait_time = backoff + random.uniform(0, backoff)
                search_results = self._search_with_ddgs(
                    keywords=keywords, max_results=max_results or 10, region=region or "wt-wt"
                )

            except WebSearchBackendUnavailable:
                # Hard fail: don't convert this to {"error": ...} since the harness should stop.
                raise
            except Exception as e:
                if "No results found" in str(e):
                    wait_time = backoff + random.uniform(0, backoff)
                    error_block = (
                        "*" * 100 + f"\n❗️❗️ [WebSearchAPI] Hit a block/rate-limit on DuckDuckGo requests. "
                        f"If unable to run eval due to repeated blocks, try to decrease job parallelism or set SERPAPI_API_KEY "
                        f"to use SerpApi's DuckDuckGo engine. Retrying in {wait_time:.1f} seconds…" + "*" * 100
                    )
                    print(error_block)
                    time.sleep(wait_time)
                    backoff = min(backoff * 2, 120)  # cap the back-off
                    continue
                else:
                    error_block = (
                        "*" * 100 + f"\n❗️❗️ [WebSearchAPI] Error from DuckDuckGo (ddgs): {str(e)}. "
                        "This is not a recognized retryable rate-limit error, so it will not be retried. "
                        "If you suspect blocking/rate limiting, consider setting SERPAPI_API_KEY to use SerpApi."
                        + "*"
                        * 100
                    )
                    print(error_block)
                return {"error": str(e)}

            break  # Success – no rate-limit error detected

        if not search_results:
            return {"error": "Failed to retrieve the search results from server. Please try again later."}

        return self._format_results(search_results[: (max_results or 10)])

    def fetch_url_content(self, url: str, mode: str = "raw") -> str:
        """
        This function retrieves content from the provided URL and processes it based on the selected mode.

        Args:
            url (str): The URL to fetch content from. Must start with 'http://' or 'https://'.
            mode (str, optional): The mode to process the fetched content. Defaults to "raw".
                Supported modes are:
                    - "raw": Returns the raw HTML content.
                    - "markdown": Converts raw HTML content to Markdown format for better readability, using html2text.
                    - "truncate": Extracts and cleans text by removing scripts, styles, and extraneous whitespace.
        """
        if not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {url}")

        try:
            # A header that mimics a browser request. This helps avoid 403 Forbidden errors.
            # TODO: Is this the best way to do this?
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/112.0.0.0 Safari/537.36"
                ),
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"
                ),
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Referer": "https://www.google.com/",
                "Sec-Fetch-Site": "same-origin",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-User": "?1",
                "Sec-Fetch-Dest": "document",
            }
            response = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
            response.raise_for_status()

            # Process the response based on the mode
            if mode == "raw":
                return {"content": response.text}

            elif mode == "markdown":
                converter = html2text.HTML2Text()
                markdown = converter.handle(response.text)
                return {"content": markdown}

            elif mode == "truncate":
                soup = BeautifulSoup(response.text, "html.parser")

                # Remove scripts and styles
                for script_or_style in soup(["script", "style"]):
                    script_or_style.extract()

                # Extract and clean text
                text = soup.get_text(separator="\n", strip=True)
                return {"content": text}
            else:
                raise ValueError(f"Unsupported mode: {mode}")

        except Exception as e:
            return {"error": f"An error occurred while fetching {url}: {str(e)}"}
