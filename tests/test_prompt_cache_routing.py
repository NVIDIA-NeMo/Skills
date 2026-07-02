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
"""Tests for BaseModel request IDs and optional prompt-cache affinity keys."""

from nemo_skills.inference.model.base import BaseModel


def test_idempotency_header_always_set_affinity_opt_in():
    # Request IDs do not depend on cache affinity.
    params = {}
    BaseModel._apply_routing_keys(params, None)
    assert "prompt_cache_key" not in params
    xid = params["extra_headers"]["X-Request-Id"]
    assert isinstance(xid, str) and len(xid) >= 16


def test_cache_key_sets_prompt_cache_key():
    params = {}
    BaseModel._apply_routing_keys(params, "conv-1")
    assert params["prompt_cache_key"] == "conv-1"
    assert params["extra_headers"]["X-Request-Id"]


def test_shared_affinity_distinct_request_ids():
    # Requests may share cache affinity without sharing request IDs.
    p1, p2 = {}, {}
    BaseModel._apply_routing_keys(p1, "conv-1")
    BaseModel._apply_routing_keys(p2, "conv-1")
    assert p1["prompt_cache_key"] == p2["prompt_cache_key"] == "conv-1"
    assert p1["extra_headers"]["X-Request-Id"] != p2["extra_headers"]["X-Request-Id"]


def test_preserves_existing_headers_and_does_not_override_request_id():
    # Existing request and custom headers are preserved.
    params = {"extra_headers": {"X-Custom": "v", "X-Request-Id": "preset"}}
    BaseModel._apply_routing_keys(params, None)
    assert params["extra_headers"]["X-Custom"] == "v"
    assert params["extra_headers"]["X-Request-Id"] == "preset"
