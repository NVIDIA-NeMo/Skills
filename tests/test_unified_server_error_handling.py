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

import importlib

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

from fastapi.testclient import TestClient

from recipes.multimodal.server.backends import (
    BackendConfig,
    GenerationRequest,
    GenerationResult,
    InferenceBackend,
    Modality,
)

unified_server = importlib.import_module("recipes.multimodal.server.unified_server")


class _ErrorBackend(InferenceBackend):
    @classmethod
    def get_config_class(cls) -> type:
        return BackendConfig

    @property
    def name(self) -> str:
        return "error_backend"

    @property
    def supported_modalities(self):
        return {Modality.TEXT}

    def load_model(self) -> None:
        self._is_loaded = True

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        return [
            GenerationResult(error=f"sensitive backend error for {request.request_id}: /tmp/secret/path")
            for request in requests
        ]


class _OkBackend(InferenceBackend):
    @classmethod
    def get_config_class(cls) -> type:
        return BackendConfig

    @property
    def name(self) -> str:
        return "ok_backend"

    @property
    def supported_modalities(self):
        return {Modality.TEXT}

    def load_model(self) -> None:
        self._is_loaded = True

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        del requests
        return [GenerationResult(text="ok")]


def test_chat_completion_does_not_leak_raw_backend_error(monkeypatch):
    monkeypatch.setattr(unified_server, "get_backend", lambda backend_type: _ErrorBackend)
    app = unified_server.create_app(
        backend_type="error_backend",
        config_dict={"model_path": "dummy"},
        batch_size=1,
        batch_timeout=0,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail.startswith("Internal server error (error_id=")
    assert "/tmp/secret/path" not in detail


def test_chat_completion_returns_500_if_audio_save_dir_cannot_be_prepared(monkeypatch):
    monkeypatch.setattr(unified_server, "get_backend", lambda backend_type: _OkBackend)
    monkeypatch.setenv("AUDIO_SAVE_DIR", "/forbidden/save/path")

    original_makedirs = unified_server.os.makedirs

    def _fail_makedirs(path, exist_ok=False):
        if path == "/forbidden/save/path":
            raise PermissionError("no write permission")
        return original_makedirs(path, exist_ok=exist_ok)

    monkeypatch.setattr(unified_server.os, "makedirs", _fail_makedirs)

    app = unified_server.create_app(
        backend_type="ok_backend",
        config_dict={"model_path": "dummy"},
        batch_size=1,
        batch_timeout=0,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hello"}]},
        )

    assert response.status_code == 500
    detail = response.json()["detail"]
    assert detail.startswith("Internal server error (error_id=")
    assert "no write permission" not in detail
