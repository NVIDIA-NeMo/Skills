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

"""Tests for unified_server audio extraction from chat-completion messages."""

import base64
import importlib

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")

extract_audio_from_messages = importlib.import_module(
    "recipes.multimodal.server.unified_server"
).extract_audio_from_messages
extract_text_from_messages = importlib.import_module(
    "recipes.multimodal.server.unified_server"
).extract_text_from_messages


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("utf-8")


def test_extract_audio_from_messages_audio_url_only():
    raw = b"audio-from-audio-url"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"data:audio/wav;base64,{_b64(raw)}"},
                }
            ],
        }
    ]

    assert extract_audio_from_messages(messages) == [raw]


def test_extract_audio_from_messages_input_audio_only():
    raw = b"audio-from-input-audio"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": _b64(raw), "format": "wav"},
                }
            ],
        }
    ]

    assert extract_audio_from_messages(messages) == [raw]


def test_extract_audio_from_messages_mixed_order_is_preserved():
    first = b"first-audio"
    second = b"second-audio"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": _b64(first), "format": "wav"},
                },
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"data:audio/wav;base64,{_b64(second)}"},
                },
            ],
        }
    ]

    assert extract_audio_from_messages(messages) == [first, second]


def test_extract_audio_from_messages_skips_non_audio_or_malformed_blocks():
    valid = b"valid-audio"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "ignore me"},
                {"type": "audio_url", "audio_url": {"url": "http://example.com/a.wav"}},
                {"type": "input_audio", "input_audio": {"format": "wav"}},
                {
                    "type": "audio_url",
                    "audio_url": {"url": f"data:audio/wav;base64,{_b64(valid)}"},
                },
            ],
        }
    ]

    assert extract_audio_from_messages(messages) == [valid]


def test_extract_text_from_messages_ignores_system_role():
    messages = [
        {"role": "system", "content": "You are a strict assistant."},
        {"role": "user", "content": "Say hello."},
        {"role": "assistant", "content": "Hello."},
    ]

    assert extract_text_from_messages(messages) == "Say hello. Hello."
