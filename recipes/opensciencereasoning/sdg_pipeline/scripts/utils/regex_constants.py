#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import re

INTERNET_PATTERNS = {
    "URL": r"https?://",
    "HTTP_LIB": r"\b(requests|urllib|urllib3|httpx|aiohttp|http\.client)\b",
    "SOCKET": r"\bimport\s+socket\b|\bsocket\.connect\(",
    "CLI": r"\b(curl|wget|aria2|lynx)\b",
    "SEARCH_API": r"\b(pubmed|ncbi|entrez|serpapi|bing|duckduckgo|google|wikipedia)\b",
    "SCRAPING": r"\b(BeautifulSoup|selenium|playwright)\b",
    "API_CLIENT": r"\b(openai|boto3|googleapiclient)\b",
}

COMPILED_INTERNET_PATTERNS = {k: re.compile(v, re.IGNORECASE) for k, v in INTERNET_PATTERNS.items()}
