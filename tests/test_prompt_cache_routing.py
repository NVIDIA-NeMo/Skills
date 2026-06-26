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
"""Unit tests for the nmux pull-queue routing keys attached by BaseModel.

Idempotency (X-Request-Id) and prefix-cache affinity (prompt_cache_key) are
DECOUPLED: X-Request-Id is per-request (distinct jobs / safe retries),
prompt_cache_key is opt-in and SHARED across requests that should co-locate.
See multiplexer docs/plans/MULTI_TURN_AFFINITY.md.
"""

from nemo_skills.inference.model.base import BaseModel


def test_idempotency_header_always_set_affinity_opt_in():
    # No cache_key → X-Request-Id present (idempotency), prompt_cache_key OMITTED
    # so the gateway infers affinity (multi-turn) / lets unique requests spread.
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
    # The branching/multi-sample invariant: same cache_key → co-locate (shared
    # prefill), but each call gets a DISTINCT X-Request-Id → distinct jobs.
    p1, p2 = {}, {}
    BaseModel._apply_routing_keys(p1, "conv-1")
    BaseModel._apply_routing_keys(p2, "conv-1")
    assert p1["prompt_cache_key"] == p2["prompt_cache_key"] == "conv-1"
    assert p1["extra_headers"]["X-Request-Id"] != p2["extra_headers"]["X-Request-Id"]


def test_preserves_existing_headers_and_does_not_override_request_id():
    # setdefault semantics: a caller-set X-Request-Id (e.g. for app-level retry
    # idempotency) and other headers survive.
    params = {"extra_headers": {"X-Custom": "v", "X-Request-Id": "preset"}}
    BaseModel._apply_routing_keys(params, None)
    assert params["extra_headers"]["X-Custom"] == "v"
    assert params["extra_headers"]["X-Request-Id"] == "preset"
