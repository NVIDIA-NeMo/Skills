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

import base64
import os
import tempfile

import pytest

from nemo_skills.inference.model import VLLMModel, get_model, models
from nemo_skills.inference.model.vllm import encode_image_to_base64, process_image_content
from nemo_skills.prompt.utils import Prompt, PromptConfig


def test_encode_image_to_base64():
    png_header = b"\x89PNG\r\n\x1a\n"
    ihdr_data = b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00"
    ihdr_crc = b"\x90wS\xde"
    iend_chunk = b"\x00\x00\x00\x00IEND\xaeB`\x82"
    minimal_png = png_header + b"\x00\x00\x00\rIHDR" + ihdr_data + ihdr_crc + iend_chunk

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(minimal_png)
        temp_path = f.name

    try:
        result = encode_image_to_base64(temp_path)
        assert result.startswith("data:image/png;base64,")
        base64_data = result.split(",")[1]
        decoded = base64.b64decode(base64_data)
        assert decoded == minimal_png
    finally:
        os.unlink(temp_path)


def test_encode_image_file_not_found():
    with pytest.raises(FileNotFoundError):
        encode_image_to_base64("/nonexistent/path/image.png")


def test_process_image_content_text_only():
    text_content = "This is a simple text prompt."
    result = process_image_content(text_content)
    assert result == text_content


def test_process_image_content_preserves_text_items():
    content = [{"type": "text", "text": "What is in this image?"}]
    result = process_image_content(content)
    assert result == content


def test_process_image_content_none():
    result = process_image_content(None)
    assert result is None


def test_process_image_content_http_url_passthrough():
    http_url = "https://example.com/image.jpg"
    content = [
        {"type": "text", "text": "Describe this image."},
        {"type": "image_url", "image_url": {"url": http_url}},
    ]
    result = process_image_content(content)
    assert result[1]["image_url"]["url"] == http_url


def test_process_image_content_data_url_passthrough():
    data_url = "data:image/png;base64,iVBORw0KGgo="
    content = [{"type": "image_url", "image_url": {"url": data_url}}]
    result = process_image_content(content)
    assert result[0]["image_url"]["url"] == data_url


def test_vllm_model_registered():
    assert "vllm" in models
    assert models["vllm"] == VLLMModel


def test_get_model_vllm():
    model = get_model(server_type="vllm", model="test-model")
    assert isinstance(model, VLLMModel)


def test_prompt_with_image_field():
    config = PromptConfig(user="{problem}", image_field="image_path", image_position="before")
    prompt = Prompt(config, tokenizer=None)
    input_dict = {"problem": "What is shown in this image?", "image_path": "path/to/image.png"}

    messages = prompt.fill(input_dict)

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    content = messages[0]["content"]
    assert isinstance(content, list)
    assert len(content) == 2
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"] == "path/to/image.png"
    assert content[1]["type"] == "text"
    assert content[1]["text"] == "What is shown in this image?"


def test_prompt_without_image_field():
    config = PromptConfig(user="{problem}")
    prompt = Prompt(config, tokenizer=None)
    input_dict = {"problem": "What is 2 + 2?"}

    messages = prompt.fill(input_dict)

    assert messages[0]["content"] == "What is 2 + 2?"


def test_prompt_image_position_after():
    config = PromptConfig(user="{problem}", image_field="image_path", image_position="after")
    prompt = Prompt(config, tokenizer=None)
    input_dict = {"problem": "Describe this.", "image_path": "img.png"}

    messages = prompt.fill(input_dict)

    content = messages[0]["content"]
    assert content[0]["type"] == "text"
    assert content[1]["type"] == "image_url"


def test_prompt_image_position_invalid():
    config = PromptConfig(user="{problem}", image_field="image_path", image_position="middle")
    prompt = Prompt(config, tokenizer=None)
    input_dict = {"problem": "Test", "image_path": "img.png"}

    with pytest.raises(ValueError, match="Invalid image_position"):
        prompt.fill(input_dict)
