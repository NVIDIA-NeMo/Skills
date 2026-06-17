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

"""Tests for audio utilities and VLLMMultimodalModel audio input handling."""

import base64
import contextlib
import os
import tempfile
from unittest.mock import patch

import pytest

from nemo_skills.inference.model.audio_utils import audio_file_to_base64
from nemo_skills.inference.model.vllm import VLLMModel
from nemo_skills.inference.model.vllm_multimodal import VLLMMultimodalModel


def test_audio_file_to_base64():
    """Test basic audio file encoding to base64."""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".wav", delete=False) as f:
        test_content = b"RIFF" + b"\x00" * 100
        f.write(test_content)
        temp_path = f.name

    try:
        result = audio_file_to_base64(temp_path)
        assert isinstance(result, str)
        assert len(result) > 0
        decoded = base64.b64decode(result)
        assert decoded == test_content
    finally:
        os.unlink(temp_path)


def _is_valid_audio_content(content_item: dict) -> bool:
    """Check if content item is a valid audio block (either format)."""
    if content_item.get("type") == "audio_url":
        url = content_item.get("audio_url", {}).get("url", "")
        return url.startswith("data:audio/wav;base64,") or url.startswith("file://")
    elif content_item.get("type") == "input_audio":
        return "data" in content_item.get("input_audio", {})
    return False


@pytest.fixture
def mock_vllm_multimodal_model(tmp_path):
    """Create a mock VLLMMultimodalModel for testing audio preprocessing."""
    with patch.object(VLLMMultimodalModel, "__init__", lambda self, **kwargs: None):
        model = VLLMMultimodalModel()
        model.data_dir = str(tmp_path)
        model.output_dir = None
        model.output_audio_dir = None
        model.enable_audio_chunking = True
        model.audio_chunk_task_types = None
        model.chunk_audio_threshold_sec = 30
        model.audio_format = "audio_url"  # Test audio_url format (for vLLM/Qwen)
        model.audio_as_path = True
        model._tunnel = None
        return model


@pytest.fixture
def mock_vllm_multimodal_model_input_audio(tmp_path):
    """Create a mock VLLMMultimodalModel configured for input_audio."""
    with patch.object(VLLMMultimodalModel, "__init__", lambda self, **kwargs: None):
        model = VLLMMultimodalModel()
        model.data_dir = str(tmp_path)
        model.output_dir = None
        model.output_audio_dir = None
        model.enable_audio_chunking = True
        model.audio_chunk_task_types = None
        model.chunk_audio_threshold_sec = 30
        model.audio_format = "input_audio"
        model.audio_as_path = False
        model._tunnel = None
        return model


def test_content_text_to_list_with_audio(mock_vllm_multimodal_model, tmp_path):
    """Test converting string content with audio to list format.

    CRITICAL: Audio must come BEFORE text for Qwen Audio to transcribe correctly.
    """
    audio_path = tmp_path / "test.wav"
    with open(audio_path, "wb") as f:
        f.write(b"RIFF" + b"\x00" * 100)

    message = {"role": "user", "content": "Describe this audio", "audio": {"path": "test.wav"}}

    result = mock_vllm_multimodal_model.content_text_to_list(message)

    assert isinstance(result["content"], list)
    assert len(result["content"]) == 2
    assert _is_valid_audio_content(result["content"][0])
    assert result["content"][0]["audio_url"]["url"] == f"file://{audio_path}"
    assert result["content"][1]["type"] == "text"


def test_content_text_to_list_with_audio_url_base64_fallback(mock_vllm_multimodal_model, tmp_path):
    """Test that local vLLM audio_url mode can still inline base64 when requested."""
    mock_vllm_multimodal_model.audio_as_path = False
    audio_path = tmp_path / "test.wav"
    with open(audio_path, "wb") as f:
        f.write(b"RIFF" + b"\x00" * 100)

    message = {"role": "user", "content": "Describe this audio", "audio": {"path": "test.wav"}}
    result = mock_vllm_multimodal_model.content_text_to_list(message)

    assert result["content"][0]["type"] == "audio_url"
    assert result["content"][0]["audio_url"]["url"].startswith("data:audio/wav;base64,")
    assert result["content"][1]["type"] == "text"


def test_content_text_to_list_with_input_audio_format(mock_vllm_multimodal_model_input_audio, tmp_path):
    """Test audio conversion with input_audio format (OpenAI native)."""
    audio_path = tmp_path / "test.wav"
    with open(audio_path, "wb") as f:
        f.write(b"RIFF" + b"\x00" * 100)

    message = {"role": "user", "content": "Describe this audio", "audio": {"path": "test.wav"}}
    result = mock_vllm_multimodal_model_input_audio.content_text_to_list(message)

    assert isinstance(result["content"], list)
    assert len(result["content"]) == 2
    # Verify input_audio format structure
    assert result["content"][0]["type"] == "input_audio"
    assert "data" in result["content"][0]["input_audio"]
    assert result["content"][0]["input_audio"]["format"] == "wav"
    assert result["content"][1]["type"] == "text"


def test_content_text_to_list_with_multiple_audios(mock_vllm_multimodal_model, tmp_path):
    """Test handling message with multiple audio files.

    CRITICAL: Audio must come BEFORE text for Qwen Audio to transcribe correctly.
    """
    audio_paths = []
    for i in range(2):
        audio_path = tmp_path / f"test_{i}.wav"
        with open(audio_path, "wb") as f:
            f.write(b"RIFF" + b"\x00" * 100)
        audio_paths.append(f"test_{i}.wav")

    message = {
        "role": "user",
        "content": "Compare these",
        "audios": [{"path": audio_paths[0]}, {"path": audio_paths[1]}],
    }

    result = mock_vllm_multimodal_model.content_text_to_list(message)

    assert isinstance(result["content"], list)
    assert len(result["content"]) == 3
    # Audio MUST come before text for Qwen Audio
    assert _is_valid_audio_content(result["content"][0])
    assert _is_valid_audio_content(result["content"][1])
    assert result["content"][2]["type"] == "text"


def test_content_text_to_list_no_audio(mock_vllm_multimodal_model):
    """Test that messages without audio are returned unchanged."""
    message = {"role": "user", "content": "Hello, world!"}
    result = mock_vllm_multimodal_model.content_text_to_list(message)

    assert result["content"] == "Hello, world!"
    assert "audio" not in result


def test_preprocess_messages_preserves_no_think(mock_vllm_multimodal_model):
    """Test that /no_think is preserved in system messages."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. /no_think"},
        {"role": "user", "content": "Hello"},
    ]

    result = mock_vllm_multimodal_model._preprocess_messages_for_model(messages)

    # /no_think should be preserved, not stripped
    assert result[0]["content"] == "You are a helpful assistant. /no_think"
    assert result[1]["content"] == "Hello"


def test_needs_audio_chunking_disabled(mock_vllm_multimodal_model):
    """Test that chunking is skipped when disabled."""
    mock_vllm_multimodal_model.enable_audio_chunking = False

    messages = [{"role": "user", "content": "Test", "audio": {"path": "test.wav"}}]
    needs_chunking, audio_path, duration = mock_vllm_multimodal_model._needs_audio_chunking(messages)

    assert needs_chunking is False
    assert audio_path is None
    assert duration == 0.0


def test_needs_audio_chunking_task_type_filter(mock_vllm_multimodal_model):
    """Test that chunking respects task type filter."""
    mock_vllm_multimodal_model.audio_chunk_task_types = ["transcription"]

    messages = [{"role": "user", "content": "Test", "audio": {"path": "test.wav"}}]

    # Task type not in filter - should not chunk
    needs_chunking, _, _ = mock_vllm_multimodal_model._needs_audio_chunking(messages, task_type="qa")
    assert needs_chunking is False

    # Task type in filter but file doesn't exist - should return False gracefully
    needs_chunking, _, _ = mock_vllm_multimodal_model._needs_audio_chunking(messages, task_type="transcription")
    assert needs_chunking is False


@contextlib.contextmanager
def _stub_vllm_base():
    """Run the real ``VLLMMultimodalModel.__init__`` while stubbing the heavy
    ``VLLMModel`` base so no server / tokenizer / network setup is needed.

    The base only needs to provide the attributes the audio-config logic reads
    (``base_url`` and ``output_dir``); everything the assertions cover lives in
    ``VLLMMultimodalModel.__init__`` itself.
    """

    def _init(self, model=None, base_url=None, **kwargs):
        self.base_url = base_url
        self.output_dir = kwargs.get("output_dir")
        self._tunnel = None  # consumed by BaseModel.__del__

    with patch.object(VLLMModel, "__init__", _init):
        yield


def test_audio_as_path_defaults_local_default_url():
    """Local vLLM defaults to audio_url but keeps file:// audio opt-in."""
    with _stub_vllm_base():
        model = VLLMMultimodalModel(base_url=None)
    assert model._external_api_mode is False
    assert model.audio_format == "audio_url"
    assert model.audio_as_path is False


def test_audio_as_path_defaults_local_explicit_localhost():
    """Explicit local URL is still treated as local without enabling file:// by default."""
    with _stub_vllm_base():
        model = VLLMMultimodalModel(base_url="http://127.0.0.1:5000/v1")
    assert model._external_api_mode is False
    assert model.audio_format == "audio_url"
    assert model.audio_as_path is False


def test_audio_as_path_defaults_external_api():
    """External API defaults to input_audio + audio_as_path=False."""
    with _stub_vllm_base():
        model = VLLMMultimodalModel(base_url="https://inference-api.nvidia.com/v1")
    assert model._external_api_mode is True
    assert model.audio_format == "input_audio"
    assert model.audio_as_path is False


def test_audio_as_path_local_opt_in():
    """audio_as_path=True is allowed for local audio_url when the server permits file:// media."""
    with _stub_vllm_base():
        model = VLLMMultimodalModel(base_url=None, audio_as_path=True)
    assert model.audio_format == "audio_url"
    assert model.audio_as_path is True


def test_audio_as_path_local_base64_explicit():
    """audio_as_path=False is allowed for local audio_url (base64 fallback)."""
    with _stub_vllm_base():
        model = VLLMMultimodalModel(base_url=None, audio_as_path=False)
    assert model.audio_format == "audio_url"
    assert model.audio_as_path is False


def test_audio_as_path_rejected_for_external_api():
    """audio_as_path=True against an external API must raise."""
    with _stub_vllm_base():
        with pytest.raises(ValueError, match="audio_as_path is only supported"):
            VLLMMultimodalModel(base_url="https://inference-api.nvidia.com/v1", audio_as_path=True)


def test_audio_as_path_rejected_for_input_audio_format():
    """audio_as_path=True with input_audio format must raise even when local."""
    with _stub_vllm_base():
        with pytest.raises(ValueError, match="audio_as_path is only supported"):
            VLLMMultimodalModel(base_url=None, audio_format="input_audio", audio_as_path=True)


def test_unsupported_audio_format_raises():
    """An unknown audio_format is rejected."""
    with _stub_vllm_base():
        with pytest.raises(ValueError, match="Unsupported audio_format"):
            VLLMMultimodalModel(base_url=None, audio_format="bogus")
