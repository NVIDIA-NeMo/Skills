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
from types import ModuleType, SimpleNamespace

from nemo_skills.inference.generate import (
    _VALIDATED_HTTPCORE_FAST_ASSIGN_VERSIONS,
    _patch_httpcore_connection_pool_assignment,
)


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


def test_httpcore_patch_skips_missing_version(monkeypatch):
    pool_cls = _install_fake_httpcore(monkeypatch, "1.0.9")
    del sys.modules["httpcore"].__version__
    original_assign = pool_cls._assign_requests_to_connections

    _patch_httpcore_connection_pool_assignment()

    assert pool_cls._assign_requests_to_connections is original_assign


def test_httpcore_patch_applies_to_validated_versions(monkeypatch):
    for version in _VALIDATED_HTTPCORE_FAST_ASSIGN_VERSIONS:
        pool_cls = _install_fake_httpcore(monkeypatch, version)
        original_assign = pool_cls._assign_requests_to_connections

        _patch_httpcore_connection_pool_assignment()

        patched_assign = pool_cls._assign_requests_to_connections
        assert patched_assign is not original_assign
        assert getattr(patched_assign, "_nemo_skills_fast_assign", False) is True


def test_httpcore_patch_skips_older_unvalidated_version(monkeypatch):
    pool_cls = _install_fake_httpcore(monkeypatch, "1.0.2")
    original_assign = pool_cls._assign_requests_to_connections

    _patch_httpcore_connection_pool_assignment()

    assert pool_cls._assign_requests_to_connections is original_assign


def test_httpcore_patch_replaces_expired_connection_in_same_pass(monkeypatch):
    pool_cls = _install_fake_httpcore(monkeypatch, "1.0.9")
    _patch_httpcore_connection_pool_assignment()

    class FakeConnection:
        def __init__(self, *, expired=False):
            self.expired = expired

        def is_closed(self):
            return False

        def has_expired(self):
            return self.expired

        def is_available(self):
            return not self.expired

        def is_idle(self):
            return not self.expired

        def can_handle_request(self, origin):
            return False

    class FakePoolRequest:
        def __init__(self):
            self.request = SimpleNamespace(url=SimpleNamespace(origin="https://example.test"))
            self.assigned_connection = None

        def is_queued(self):
            return self.assigned_connection is None

        def assign_to_connection(self, connection):
            self.assigned_connection = connection

    expired_connection = FakeConnection(expired=True)
    new_connection = FakeConnection()
    pool_request = FakePoolRequest()
    pool = SimpleNamespace(
        _http2=False,
        _connections=[expired_connection],
        _requests=[pool_request],
        _max_connections=1,
        _max_keepalive_connections=1,
        create_connection=lambda origin: new_connection,
    )

    closing_connections = pool_cls._assign_requests_to_connections(pool)

    assert closing_connections == [expired_connection]
    assert pool_request.assigned_connection is new_connection
    assert pool._connections == [new_connection]
