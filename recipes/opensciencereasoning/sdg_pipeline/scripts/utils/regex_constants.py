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
