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

import asyncio
import importlib

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

unified_server = importlib.import_module("recipes.multimodal.server.unified_server")
GenerationRequest = importlib.import_module("recipes.multimodal.server.backends").GenerationRequest


class _MismatchedBackend:
    def generate(self, requests):
        del requests
        return []


def test_request_batcher_fails_on_batch_result_length_mismatch():
    async def _run():
        batcher = unified_server.RequestBatcher(_MismatchedBackend(), batch_size=1, batch_timeout=0)
        with pytest.raises(RuntimeError, match="Backend returned 0 results for 1 requests"):
            await batcher.add_request(GenerationRequest(text="hello"))

    asyncio.run(_run())
