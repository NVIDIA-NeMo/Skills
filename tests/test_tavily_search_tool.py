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

from __future__ import annotations

import asyncio
import json

import pytest


@pytest.fixture
def exclude_domains_config(tmp_path):
    exclude_file = tmp_path / "exclude.json"
    exclude_file.write_text(
        json.dumps({"notices": [{"properties": [{"type": "domain", "value": "blocked.example"}]}]}),
        encoding="utf-8",
    )
    return str(exclude_file)


def test_direct_tavily_search_tool_uses_configured_hidden_args(exclude_domains_config):
    async def run_test():
        from nemo_skills.mcp.servers.tavily_search_tool import DirectTavilySearchTool

        tool = DirectTavilySearchTool()
        tool.configure(
            {
                "tavily_api_key": "test-key",
                "exclude_domains_config": exclude_domains_config,
                "num_results": 3,
            }
        )

        calls = []

        async def fake_post_json(endpoint, payload):
            calls.append((endpoint, payload))
            return {"answer": "Paris", "results": []}

        tool._post_json = fake_post_json

        tools = await tool.list_tools()
        assert tools[0]["name"] == "web_search"
        assert set(tools[0]["input_schema"]["properties"]) == {"query"}
        assert tools[0]["input_schema"]["additionalProperties"] is False

        invalid = await tool.execute("web_search", {"query": "capital of France", "typo": True})
        assert invalid == {"error": "Unsupported argument: typo"}

        result = await tool.execute(
            "web_search",
            {"query": "capital of France", "num_results": 99, "answer_type": "results"},
            extra_args={"request_id": "req-search"},
        )

        assert result == "Paris"
        assert calls == [
            (
                "/search",
                {
                    "query": "capital of France",
                    "search_depth": "basic",
                    "include_answer": "basic",
                    "max_results": 3,
                    "exclude_domains": ["blocked.example"],
                },
            )
        ]
        assert len(tool.requests_to_metrics["req-search"].async_tavily_calls) == 1
        assert tool.requests_to_metrics["req-search"].async_tavily_calls[0].function == "search"

    asyncio.run(run_test())


def test_direct_tavily_gym_tool_web_search_formats_results_and_hides_internal_args(exclude_domains_config):
    async def run_test():
        from nemo_skills.mcp.servers.tavily_search_tool import DirectTavilyGymTool

        tool = DirectTavilyGymTool()
        tool.configure({"tavily_api_key": "test-key", "exclude_domains_config": exclude_domains_config})

        calls = []

        async def fake_post_json(endpoint, payload):
            calls.append((endpoint, payload))
            return {
                "results": [
                    {
                        "url": "https://example.com/page1",
                        "title": "Example Page 1",
                        "content": "This is the content of page 1",
                        "score": 0.95,
                        "raw_content": "raw content",
                    }
                ]
            }

        tool._post_json = fake_post_json

        tools = await tool.list_tools()
        assert {t["name"] for t in tools} == {"web_search", "find_in_page", "scroll_page"}
        assert all(t["input_schema"]["additionalProperties"] is False for t in tools)

        invalid = await tool.execute("web_search", {"query": "NVIDIA GPU programming", "typo": True})
        assert invalid == {"error": "Unsupported argument: typo"}

        result = await tool.execute(
            "web_search",
            {"query": "NVIDIA GPU programming", "max_results": 1, "exclude_domains": ["model.example"]},
            extra_args={"request_id": "req-gym-search"},
        )

        assert "Search Results" in result
        assert "[1] Example Page 1 (example.com)" in result
        assert "URL: https://example.com/page1" in result
        assert "This is the content of page 1" in result
        assert "0.95" not in result
        assert "raw content" not in result
        assert calls == [
            (
                "/search",
                {
                    "query": "NVIDIA GPU programming",
                    "max_results": 10,
                    "exclude_domains": ["blocked.example"],
                    "search_depth": "advanced",
                },
            )
        ]

    asyncio.run(run_test())


def test_direct_tavily_gym_tool_find_in_page_and_scroll_page(exclude_domains_config):
    async def run_test():
        from nemo_skills.mcp.servers.tavily_search_tool import DirectTavilyGymTool

        tool = DirectTavilyGymTool()
        tool.configure(
            {
                "tavily_api_key": "test-key",
                "exclude_domains_config": exclude_domains_config,
                "max_result_chars": 35,
            }
        )

        assert await tool.execute("find_in_page", {"url": None, "query": "x"}) == {"error": "URL is none"}
        assert await tool.execute("find_in_page", {"url": "https://sub.blocked.example/a", "query": "x"}) == (
            {"error": "URL is in excluded domains"}
        )
        assert await tool.execute(
            "scroll_page",
            {"url": "https://example.com/page", "start_index": 0, "n": "many"},
        ) == {"error": "Invalid n: n must be an integer"}

        calls = []

        async def fake_post_json(endpoint, payload):
            calls.append((endpoint, payload))
            if payload.get("query"):
                return {
                    "results": [
                        {
                            "url": payload["urls"],
                            "raw_content": "Hello [edit] world\n[Jump to content]\nContent here\nMore content after limit",
                        }
                    ]
                }
            return {"results": [{"url": payload["urls"], "raw_content": "zero one two three four five"}]}

        tool._post_json = fake_post_json

        find_result = await tool.execute(
            "find_in_page",
            {"url": "https://example.com/page", "query": "content"},
            extra_args={"request_id": "req-find"},
        )
        assert "Content from: example.com" in find_result
        assert 'Query: "content"' in find_result
        assert "[edit]" not in find_result
        assert "[Jump to content]" not in find_result
        assert "[...truncated, use scroll_page for full content]" in find_result

        scroll_result = await tool.execute(
            "scroll_page", {"url": "https://example.com/page", "start_index": 2, "n": 3}
        )
        assert "Showing words [2-5] of 6" in scroll_result
        assert "L0: two three four" in scroll_result

        cached_scroll_result = await tool.execute(
            "scroll_page", {"url": "https://example.com/page", "start_index": 0, "n": 1}
        )
        assert "Showing words [0-1] of 6" in cached_scroll_result

        extract_calls = [call for call in calls if call[0] == "/extract"]
        assert len(extract_calls) == 2

    asyncio.run(run_test())


def test_direct_tavily_gym_tool_scroll_cache_is_bounded(exclude_domains_config):
    async def run_test():
        from nemo_skills.mcp.servers.tavily_search_tool import DirectTavilyGymTool

        tool = DirectTavilyGymTool()
        tool.configure(
            {"tavily_api_key": "test-key", "exclude_domains_config": exclude_domains_config, "max_cached_pages": 1}
        )

        calls = []

        async def fake_post_json(endpoint, payload):
            calls.append((endpoint, payload["urls"]))
            return {"results": [{"url": payload["urls"], "raw_content": f"{payload['urls']} content"}]}

        tool._post_json = fake_post_json

        await tool.execute("scroll_page", {"url": "https://example.com/a"})
        await tool.execute("scroll_page", {"url": "https://example.com/b"})
        await tool.execute("scroll_page", {"url": "https://example.com/a"})

        assert calls == [
            ("/extract", "https://example.com/a"),
            ("/extract", "https://example.com/b"),
            ("/extract", "https://example.com/a"),
        ]
        assert list(tool._page_cache) == ["https://example.com/a"]

    asyncio.run(run_test())


def test_direct_tavily_gym_tool_scroll_cache_ignores_tracking_params(exclude_domains_config):
    async def run_test():
        from nemo_skills.mcp.servers.tavily_search_tool import DirectTavilyGymTool

        tool = DirectTavilyGymTool()
        tool.configure({"tavily_api_key": "test-key", "exclude_domains_config": exclude_domains_config})

        calls = []

        async def fake_post_json(endpoint, payload):
            calls.append((endpoint, payload["urls"]))
            return {"results": [{"url": payload["urls"], "raw_content": "same page content"}]}

        tool._post_json = fake_post_json

        await tool.execute("scroll_page", {"url": "https://example.com/page?id=1&utm_source=a#section"})
        await tool.execute("scroll_page", {"url": "https://example.com/page?utm_campaign=b&id=1"})
        await tool.execute("scroll_page", {"url": "https://example.com/page?id=2&utm_source=a"})

        assert calls == [
            ("/extract", "https://example.com/page?id=1&utm_source=a#section"),
            ("/extract", "https://example.com/page?id=2&utm_source=a"),
        ]
        assert list(tool._page_cache) == ["https://example.com/page?id=1", "https://example.com/page?id=2"]

    asyncio.run(run_test())


def test_direct_tavily_tool_loads_exclude_domains_config(tmp_path):
    from nemo_skills.mcp.servers.tavily_search_tool import DirectTavilySearchTool

    exclude_file = tmp_path / "exclude.json"
    exclude_file.write_text(
        json.dumps({"notices": [{"properties": [{"type": "domain", "value": "blocked.example"}]}]}),
        encoding="utf-8",
    )

    tool = DirectTavilySearchTool()
    tool.configure({"tavily_api_key": "test-key", "exclude_domains_config": str(exclude_file)})

    assert tool._exclude_domains == ["blocked.example"]


def test_direct_tavily_tool_requires_exclude_domains_config():
    from nemo_skills.mcp.servers.tavily_search_tool import DirectTavilySearchTool, TavilySearchTool

    tool = DirectTavilySearchTool()
    assert "exclude_domains_config" not in tool.default_config()
    assert "exclude_domains_config" not in TavilySearchTool().default_config()

    with pytest.raises(ValueError, match="exclude_domains_config is required"):
        tool.configure({"tavily_api_key": "test-key", "exclude_domains": ["blocked.example"]})

    with pytest.raises(ValueError, match="Unknown DirectTavilySearchTool override"):
        tool.configure({"tavily_api_key": "test-key", "require_exclude_domains_config": False})


def test_direct_tavily_tool_rotates_api_keys(exclude_domains_config):
    from nemo_skills.mcp.servers.tavily_search_tool import DirectTavilyGymTool

    tool = DirectTavilyGymTool()
    tool.configure({"tavily_api_key": ["key-a", "key-b"], "exclude_domains_config": exclude_domains_config})

    assert [tool._select_api_key(), tool._select_api_key(), tool._select_api_key()] == ["key-a", "key-b", "key-a"]


def test_direct_tavily_browser_search_memory_open_result_and_metrics(exclude_domains_config):
    async def run_test():
        from nemo_skills.mcp.servers.tavily_search_tool import DirectTavilyBrowserTool

        tool = DirectTavilyBrowserTool()
        tool.configure(
            {
                "tavily_api_key": "test-key",
                "exclude_domains_config": exclude_domains_config,
                "max_open_result_chars": 80,
            }
        )

        calls = []

        async def fake_post_json(endpoint, payload):
            calls.append((endpoint, payload))
            if endpoint == "/search":
                return {
                    "results": [
                        {
                            "url": "https://example.com/page1",
                            "title": "Example Page 1",
                            "content": "Snippet one",
                        },
                        {
                            "url": "https://example.org/page2",
                            "title": "Example Page 2",
                            "content": "Snippet two",
                        },
                    ]
                }
            return {
                "results": [
                    {
                        "url": payload["urls"],
                        "raw_content": "Alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu.",
                    }
                ]
            }

        tool._post_json = fake_post_json

        search_result = await tool.execute(
            "web_search", {"query": "example query"}, extra_args={"request_id": "req-browser"}
        )
        assert "[1] Example Page 1 (example.com)" in search_result
        assert "[2] Example Page 2 (example.org)" in search_result

        opened = await tool.execute("open_result", {"index": 2}, extra_args={"request_id": "req-browser"})
        assert "Opened search result [2]: Example Page 2" in opened
        assert "URL: https://example.org/page2" in opened
        assert "L0: Alpha beta gamma" in opened

        metrics = await tool.get_request_metrics("req-browser")
        assert metrics["web_search_calls"] == 1
        assert metrics["open_result_calls"] == 1
        assert metrics["search_results_in_memory"] == 2
        assert metrics["cached_pages"] == 1
        assert metrics["cache_misses"] == 1
        assert metrics["tavily_http_calls"] == 2
        assert metrics["tavily_http_calls_by_function"] == {"search": 1, "extract": 1}

        await tool.cleanup_request("req-browser")
        assert await tool.get_request_metrics("req-browser") == {}

    asyncio.run(run_test())


def test_direct_tavily_browser_scroll_page_is_bounded_and_request_cached(exclude_domains_config):
    async def run_test():
        from nemo_skills.mcp.servers.tavily_search_tool import DirectTavilyBrowserTool

        tool = DirectTavilyBrowserTool()
        tool.configure(
            {
                "tavily_api_key": "test-key",
                "exclude_domains_config": exclude_domains_config,
                "max_scroll_words": 3,
                "max_scroll_chars": 100,
            }
        )

        calls = []

        async def fake_post_json(endpoint, payload):
            calls.append((endpoint, payload))
            return {
                "results": [
                    {
                        "url": payload["urls"],
                        "raw_content": "zero one two three four five six seven eight nine",
                    }
                ]
            }

        tool._post_json = fake_post_json

        invalid = await tool.execute(
            "scroll_page",
            {"url": "https://example.com/page", "start_index": 0, "n": "many"},
            extra_args={"request_id": "req-scroll"},
        )
        assert invalid == {"error": "Invalid n: n must be an integer"}

        first = await tool.execute(
            "scroll_page",
            {"url": "https://example.com/page", "start_index": -5, "n": 10},
            extra_args={"request_id": "req-scroll"},
        )
        assert "Showing words [0-3] of 10" in first
        assert "Requested window capped at 3 words" in first
        assert "L0: zero one two" in first

        second = await tool.execute(
            "scroll_page",
            {"url": "https://example.com/page", "start_index": 2, "n": 2},
            extra_args={"request_id": "req-scroll"},
        )
        assert "Showing words [2-4] of 10" in second
        assert "L0: two three" in second

        beyond = await tool.execute(
            "scroll_page",
            {"url": "https://example.com/page", "start_index": 99, "n": 2},
            extra_args={"request_id": "req-scroll"},
        )
        assert "Start index is beyond page length." in beyond

        extract_calls = [call for call in calls if call[0] == "/extract"]
        assert len(extract_calls) == 1

        metrics = await tool.get_request_metrics("req-scroll")
        assert metrics["scroll_page_calls"] == 4
        assert metrics["tool_errors"] == 1
        assert metrics["cache_misses"] == 1
        assert metrics["cache_hits"] == 2

    asyncio.run(run_test())


def test_direct_tavily_browser_scroll_cache_ignores_tracking_params(exclude_domains_config):
    async def run_test():
        from nemo_skills.mcp.servers.tavily_search_tool import DirectTavilyBrowserTool

        tool = DirectTavilyBrowserTool()
        tool.configure({"tavily_api_key": "test-key", "exclude_domains_config": exclude_domains_config})

        calls = []

        async def fake_post_json(endpoint, payload):
            calls.append((endpoint, payload["urls"]))
            return {"results": [{"url": payload["urls"], "raw_content": "zero one two"}]}

        tool._post_json = fake_post_json

        first = await tool.execute(
            "scroll_page",
            {"url": "https://example.com/page?id=1&utm_source=newsletter#intro"},
            extra_args={"request_id": "req-tracking"},
        )
        second = await tool.execute(
            "scroll_page",
            {"url": "https://example.com/page?utm_medium=email&id=1"},
            extra_args={"request_id": "req-tracking"},
        )

        assert "L0: zero one two" in first
        assert "L0: zero one two" in second
        assert calls == [("/extract", "https://example.com/page?id=1&utm_source=newsletter#intro")]

        metrics = await tool.get_request_metrics("req-tracking")
        assert metrics["cache_misses"] == 1
        assert metrics["cache_hits"] == 1
        assert metrics["cached_pages"] == 1

    asyncio.run(run_test())


def test_direct_tavily_v3_name_remains_compatible():
    from nemo_skills.mcp.servers.tavily_search_tool import DirectTavilyBrowserTool, DirectTavilyV3Tool

    assert issubclass(DirectTavilyV3Tool, DirectTavilyBrowserTool)
    assert DirectTavilyV3Tool().__class__.__name__ == "DirectTavilyV3Tool"


def test_is_url_substring_blocked_normalizes_and_is_page_precise():
    from nemo_skills.mcp.servers.tavily_search_tool import _is_url_substring_blocked, _normalize_url_for_substring

    assert _normalize_url_for_substring("GitHub.com/Foo/Bar") == "github.comfoobar"

    # Page-precise: block the HLE paper / specific repo, but NOT the rest of arxiv / github.
    patterns = ["2501.14249", "github.com/centerforaisafety/hle", "://scale.com", ".scale.com", "last-exam"]
    assert _is_url_substring_blocked("https://arxiv.org/abs/2501.14249", patterns)
    assert not _is_url_substring_blocked("https://arxiv.org/abs/1706.03762", patterns)
    assert _is_url_substring_blocked("https://github.com/centerforaisafety/hle", patterns)
    assert not _is_url_substring_blocked("https://github.com/pytorch/pytorch", patterns)
    # Anchored "://scale.com" matches scale.com but not descale.com.
    assert _is_url_substring_blocked("https://scale.com/leaderboard", patterns)
    assert not _is_url_substring_blocked("https://descale.com/page", patterns)
    # Plain substring.
    assert _is_url_substring_blocked("https://blog.example/the-last-exam-answers", patterns)
    # No patterns -> nothing blocked.
    assert not _is_url_substring_blocked("https://arxiv.org/abs/2501.14249", [])


@pytest.fixture
def exclude_url_substrings_config(tmp_path):
    exclude_file = tmp_path / "exclude_substr.json"
    exclude_file.write_text(
        json.dumps(
            {
                "notices": [
                    {
                        "properties": [
                            {"type": "domain", "value": "blocked.example"},
                            {"type": "url_substring", "value": "2501.14249"},
                            {"type": "url_substring", "value": "github.com/centerforaisafety/hle"},
                        ]
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return str(exclude_file)


def test_tool_loads_exclude_url_substrings_from_config(exclude_url_substrings_config):
    from nemo_skills.mcp.servers.tavily_search_tool import DirectTavilySearchTool

    tool = DirectTavilySearchTool()
    tool.configure({"tavily_api_key": "test-key", "exclude_domains_config": exclude_url_substrings_config})

    assert tool._exclude_domains == ["blocked.example"]
    assert tool._exclude_url_substrings == ["2501.14249", "github.com/centerforaisafety/hle"]
    # Substring patterns and the (separate) domain list both feed _is_url_blocked.
    assert tool._is_url_blocked("https://arxiv.org/abs/2501.14249")
    assert tool._is_url_blocked("https://sub.blocked.example/x")
    assert not tool._is_url_blocked("https://arxiv.org/abs/1706.03762")


def test_browser_web_search_filters_url_substring_results(exclude_url_substrings_config):
    async def run_test():
        from nemo_skills.mcp.servers.tavily_search_tool import DirectTavilyBrowserTool

        tool = DirectTavilyBrowserTool()
        tool.configure({"tavily_api_key": "test-key", "exclude_domains_config": exclude_url_substrings_config})

        async def fake_post_json(endpoint, payload):
            return {
                "results": [
                    {"url": "https://arxiv.org/abs/2501.14249", "title": "HLE Paper", "content": "leak"},
                    {"url": "https://arxiv.org/abs/1706.03762", "title": "Attention Is All You Need", "content": "ok"},
                ]
            }

        tool._post_json = fake_post_json

        result = await tool.execute("web_search", {"query": "humanity last exam"}, extra_args={"request_id": "req"})
        assert "Attention Is All You Need" in result
        assert "HLE Paper" not in result
        assert "2501.14249" not in result

        # Opening / extracting the blocked page is refused too.
        blocked = await tool.execute(
            "find_in_page",
            {"url": "https://arxiv.org/abs/2501.14249", "query": "answer"},
            extra_args={"request_id": "req"},
        )
        assert "excluded domains" in str(blocked)

    asyncio.run(run_test())


def test_gym_web_search_filters_url_substring_results(exclude_url_substrings_config):
    async def run_test():
        from nemo_skills.mcp.servers.tavily_search_tool import DirectTavilyGymTool

        tool = DirectTavilyGymTool()
        tool.configure({"tavily_api_key": "test-key", "exclude_domains_config": exclude_url_substrings_config})

        async def fake_post_json(endpoint, payload):
            return {
                "results": [
                    {"url": "https://github.com/centerforaisafety/hle", "title": "HLE repo", "content": "leak"},
                    {"url": "https://example.com/ok", "title": "Allowed", "content": "fine"},
                ]
            }

        tool._post_json = fake_post_json

        result = await tool.execute("web_search", {"query": "hle"}, extra_args={"request_id": "req"})
        assert "Allowed" in result
        assert "HLE repo" not in result
        assert "centerforaisafety" not in result

    asyncio.run(run_test())
