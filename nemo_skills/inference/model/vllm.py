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

import logging

import requests

from nemo_skills.utils import get_logger_name

from .base import BaseModel
from .utils import ServerTokenizer

LOG = logging.getLogger(get_logger_name(__file__))


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
    ) -> dict:
        assert reasoning_effort is None, "reasoning_effort is not supported for text completion requests"
        assert tools is None, "tools are not supported for text completion requests"
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
    ) -> dict:
        request = {
            "messages": messages,
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
