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
import logging
import mimetypes
import os
from pathlib import Path

import requests

from nemo_skills.utils import get_logger_name

from .base import BaseModel
from .utils import ServerTokenizer

LOG = logging.getLogger(get_logger_name(__file__))


def encode_image_to_base64(image_path: str) -> str:
    """Encode a local image file to base64 data URL."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    mime_type, _ = mimetypes.guess_type(str(path))
    if mime_type is None:
        mime_type = "image/jpeg"

    with open(path, "rb") as f:
        image_data = f.read()

    base64_data = base64.b64encode(image_data).decode("utf-8")
    return f"data:{mime_type};base64,{base64_data}"


def process_image_content(content: list | str | None, data_dir: str = "") -> list | str | None:
    """Process message content to handle image paths and URLs.

    Converts local file paths to base64 data URLs if needed.
    HTTP/HTTPS URLs and existing data URLs are passed through unchanged.
    """
    if content is None or isinstance(content, str):
        return content

    processed_content = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "image_url":
            image_url = item.get("image_url", {})
            url = image_url.get("url", "")

            if url and not url.startswith(("data:", "http://", "https://")):
                if not os.path.isabs(url) and data_dir:
                    resolved_path = os.path.join(data_dir, url)
                else:
                    resolved_path = url

                try:
                    base64_url = encode_image_to_base64(resolved_path)
                    processed_item = {
                        "type": "image_url",
                        "image_url": {"url": base64_url},
                    }
                    for key in image_url:
                        if key != "url":
                            processed_item["image_url"][key] = image_url[key]
                    item = processed_item
                except FileNotFoundError:
                    LOG.error(
                        f"Image file not found: {resolved_path} "
                        f"(original path: {url}, data_dir: {data_dir or 'not set'})"
                    )
                    raise

            processed_content.append(item)
        else:
            processed_content.append(item)

    return processed_content


class VLLMModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_tokenizer_endpoint(self):
        """
        Returns the tokenizer endpoint if available, otherwise returns None.
        """
        tokenize_url = self.base_url.replace("/v1", "/tokenize")
        payload = {"prompt": "Test prompt"}

        try:
            response = requests.post(tokenize_url, json=payload)
            if response.status_code == 200:
                LOG.info(f"Tokenize endpoint is available! - {tokenize_url}")
                return ServerTokenizer(tokenize_url)
            else:
                return None
        except requests.exceptions.RequestException:
            return None

    def _build_request_body(self, top_k, min_p, repetition_penalty, extra_body: dict = None):
        full_extra_body = {
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "spaces_between_special_tokens": False,
        }

        if top_k > 0:
            full_extra_body["top_k"] = top_k

        if extra_body:
            full_extra_body.update(extra_body)

        return full_extra_body

    def _build_completion_request_params(
        self,
        prompt: str,
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = -1,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = None,
        top_logprobs: int | None = None,
        timeout: int | None = None,
        stop_phrases: list[str] | None = None,
        stream: bool = False,
        reasoning_effort: str | None = None,
        extra_body: dict = None,
        tools: list[dict] | None = None,
        response_format=None,
    ) -> dict:
        assert reasoning_effort is None, "reasoning_effort is not supported for text completion requests"
        assert tools is None, "tools are not supported for text completion requests"
        assert response_format is None, "response_format is not supported for text completion requests"
        return {
            "prompt": prompt,
            "max_tokens": tokens_to_generate,
            "temperature": temperature,
            "top_p": top_p,
            "seed": random_seed,
            "stop": stop_phrases or None,
            "logprobs": top_logprobs,
            "stream": stream,
            "echo": False,
            "skip_special_tokens": False,
            "n": 1,
            "logit_bias": None,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "timeout": timeout,
            "extra_body": self._build_request_body(top_k, min_p, repetition_penalty, extra_body=extra_body),
        }

    def _build_chat_request_params(
        self,
        messages: list[dict],
        stream: bool,
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = -1,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: list[str] | None = None,
        timeout: int | None = None,
        top_logprobs: int | None = None,
        reasoning_effort: str | None = None,
        tools: list[dict] | None = None,
        extra_body: dict = None,
        response_format=None,
    ) -> dict:
        # Process messages to handle image content (VLM support)
        processed_messages = []
        for msg in messages:
            processed_msg = msg.copy()
            if "content" in processed_msg:
                processed_msg["content"] = process_image_content(processed_msg["content"], self.data_dir)
            processed_messages.append(processed_msg)

        request = {
            "messages": processed_messages,
            "max_tokens": tokens_to_generate,
            "temperature": temperature,
            "top_p": top_p,
            "seed": random_seed,
            "stop": stop_phrases or None,
            "logprobs": top_logprobs is not None,
            "top_logprobs": top_logprobs,
            "n": 1,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "stream": stream,
            "timeout": timeout,
            "extra_body": self._build_request_body(top_k, min_p, repetition_penalty, extra_body=extra_body),
            "tools": tools,
            "response_format": response_format,
        }
        if reasoning_effort:
            request["allowed_openai_params"] = ["reasoning_effort"]
            request["reasoning_effort"] = reasoning_effort
        return request

    def _build_responses_request_params(self, input, **kwargs) -> dict:
        # Parameters are the same as chat completion request params
        # For now, we hack this by renaming messages to input
        # Until we need more parameters for responses API
        responses_params = self._build_chat_request_params(messages=input, **kwargs)
        responses_params["input"] = responses_params.pop("messages")
        return responses_params
