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

from __future__ import annotations

import asyncio
import json


def test_direct_tavily_search_tool_uses_configured_hidden_args():
    async def run_test():
        from nemo_skills.mcp.servers.tavily_search_tool import DirectTavilySearchTool

        tool = DirectTavilySearchTool()
        tool.configure(
            {
                "tavily_api_key": "test-key",
                "exclude_domains": ["blocked.example"],
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


def test_direct_tavily_gym_tool_web_search_formats_results_and_hides_internal_args():
    async def run_test():
        from nemo_skills.mcp.servers.tavily_search_tool import DirectTavilyGymTool

        tool = DirectTavilyGymTool()
        tool.configure({"tavily_api_key": "test-key", "exclude_domains": ["blocked.example"]})

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


def test_direct_tavily_gym_tool_find_in_page_and_scroll_page():
    async def run_test():
        from nemo_skills.mcp.servers.tavily_search_tool import DirectTavilyGymTool

        tool = DirectTavilyGymTool()
        tool.configure(
            {
                "tavily_api_key": "test-key",
                "exclude_domains": ["blocked.example"],
                "max_result_chars": 35,
            }
        )

        assert await tool.execute("find_in_page", {"url": None, "query": "x"}) == "URL is none"
        assert await tool.execute("find_in_page", {"url": "https://sub.blocked.example/a", "query": "x"}) == (
            "URL is in excluded domains"
        )

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


def test_direct_tavily_tool_rotates_api_keys():
    from nemo_skills.mcp.servers.tavily_search_tool import DirectTavilyGymTool

    tool = DirectTavilyGymTool()
    tool.configure({"tavily_api_key": ["key-a", "key-b"]})

    assert [tool._select_api_key(), tool._select_api_key(), tool._select_api_key()] == ["key-a", "key-b", "key-a"]
