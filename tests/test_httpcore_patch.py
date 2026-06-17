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

"""Tests for the optional httpcore connection-pool hot-path patch."""

import sys
from types import ModuleType

from nemo_skills.inference.generate import _patch_httpcore_connection_pool_assignment


def _install_fake_httpcore(monkeypatch, version: str):
    httpcore_mod = ModuleType("httpcore")
    httpcore_mod.__version__ = version
    httpcore_mod.__path__ = []
    async_mod = ModuleType("httpcore._async")
    async_mod.__path__ = []
    pool_mod = ModuleType("httpcore._async.connection_pool")

    class AsyncConnectionPool:
        def _assign_requests_to_connections(self):
            return ["original"]

    pool_mod.AsyncConnectionPool = AsyncConnectionPool
    httpcore_mod._async = async_mod
    async_mod.connection_pool = pool_mod

    monkeypatch.setitem(sys.modules, "httpcore", httpcore_mod)
    monkeypatch.setitem(sys.modules, "httpcore._async", async_mod)
    monkeypatch.setitem(sys.modules, "httpcore._async.connection_pool", pool_mod)
    return AsyncConnectionPool


def test_httpcore_patch_skips_unvalidated_version(monkeypatch):
    pool_cls = _install_fake_httpcore(monkeypatch, "1.0.10")
    original_assign = pool_cls._assign_requests_to_connections

    _patch_httpcore_connection_pool_assignment()

    assert pool_cls._assign_requests_to_connections is original_assign


def test_httpcore_patch_applies_to_validated_version(monkeypatch):
    pool_cls = _install_fake_httpcore(monkeypatch, "1.0.9")
    original_assign = pool_cls._assign_requests_to_connections

    _patch_httpcore_connection_pool_assignment()

    patched_assign = pool_cls._assign_requests_to_connections
    assert patched_assign is not original_assign
    assert getattr(patched_assign, "_nemo_skills_fast_assign", False) is True
