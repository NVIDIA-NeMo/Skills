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

import os

from .base import BaseModel


class GeminiModel(BaseModel):
    MODEL_PROVIDER = "gemini"

    def __init__(self, base_url: str | None = None, *args, **kwargs):
        """
        model:
            - gemini-2.5-pro: thinking budget 128-32768 (default: no thinking, we should enable thinking to prevent errors.)
            - gemini-2.5-flash: thinking budget 0-24576 (default: no thinking)
            - gemini-2.5-flash-lite: thinking budget 0-24576 (default: no thinking)
        """
        # Use empty string only if no base_url provided (uses litellm's default Gemini endpoint)
        # Otherwise pass through the custom base_url
        super().__init__(base_url=base_url if base_url else "", *args, **kwargs)

    def _get_api_key(self, api_key: str | None, api_key_env_var: str | None, base_url: str) -> str | None:
        api_key = super()._get_api_key(api_key, api_key_env_var, base_url)

        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY is required for Gemini models and could not be found.")
        return api_key

    def _build_chat_request_params(
        self,
        messages: list[dict],
        tokens_to_generate: int,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        repetition_penalty: float,
        random_seed: int,
        stop_phrases: list[str],
        timeout: int | None,
        top_logprobs: int | None,
        stream: bool,
        reasoning_effort: str | None,
        extra_body: dict = None,
        tools: list[dict] | None = None,
        response_format=None,
    ) -> dict:
        """
        https://github.com/BerriAI/litellm/blob/v1.75.0-nightly/litellm/constants.py#L45-L56
        reasoning_effort:
            - None: thinking budget tokens: 0
            - low: maximum thinking budget tokens: 1024 (env var: DEFAULT_REASONING_EFFORT_LOW_THINKING_BUDGET)
            - medium: maximum thinking budget tokens: 2048 (env var: DEFAULT_REASONING_EFFORT_MEDIUM_THINKING_BUDGET)
            - high: maximum thinking budget tokens: 4096 (env var: DEFAULT_REASONING_EFFORT_HIGH_THINKING_BUDGET)
            - dynamic: maximum thinking budget tokens: -1
        """
        assert min_p == 0.0, "`min_p` is not supported by Gemini API, please set it to 0.0."
        assert repetition_penalty == 1.0, (
            "`repetition_penalty` is not supported by Gemini API, please set it to default value `1.0`."
        )
        assert not extra_body, "`extra_body` is not supported by Gemini API, please set it to None or empty dict"
        if response_format is not None:
            raise NotImplementedError()

        # Vertext AI params: https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/inference
        # litellm default params: https://github.com/BerriAI/litellm/blob/v1.75.0-nightly/litellm/llms/gemini/chat/transformation.py#L73-L90
        # litellm other params: https://github.com/BerriAI/litellm/blob/v1.75.0-nightly/litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py#L147-L174
        params = {
            "messages": messages,
            "stop": stop_phrases or None,
            "timeout": timeout,
            "stream": stream,
            "tools": tools,
            "max_completion_tokens": tokens_to_generate,
            "temperature": temperature,
            "top_p": top_p,
            "logprobs": top_logprobs is not None,
            "top_k": top_k if top_k > 0 else None,
            "seed": random_seed,
            "top_logprobs": top_logprobs,
            "allowed_openai_params": ["top_k", "seed", "top_logprobs"],
        }

        if reasoning_effort is None:
            # https://github.com/BerriAI/litellm/blob/v1.75.0-nightly/litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py#L438-L442
            reasoning_effort = "disable"

        elif reasoning_effort == "dynamic":
            reasoning_effort = None
            # https://github.com/BerriAI/litellm/blob/v1.75.0-nightly/litellm/llms/vertex_ai/gemini/vertex_and_google_ai_studio_gemini.py#L451-L465
            params["thinking"] = {
                "type": "enabled",
                "budget_tokens": -1,
                "include_thoughts": True,
            }
            params["allowed_openai_params"].append("thinking")

        params["reasoning_effort"] = reasoning_effort

        return params
