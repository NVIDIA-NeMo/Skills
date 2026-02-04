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

import copy
import os
import re

from .base import BaseModel


class OpenAIModel(BaseModel):
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: str = "5000",
        model: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
        **kwargs,
    ):
        model = model or os.getenv("NEMO_SKILLS_OPENAI_MODEL")
        self.model = model
        if model is None:
            raise ValueError("model argument is required for OpenAI model.")

        if base_url is None:
            base_url = os.getenv("NEMO_SKILLS_OPENAI_BASE_URL", f"http://{host}:{port}/v1")

        super().__init__(
            model=model,
            base_url=base_url,
            max_retries=max_retries,
            **kwargs,
        )

    def _get_api_key(self, api_key: str | None, api_key_env_var: str | None, base_url: str) -> str | None:
        api_key = super()._get_api_key(api_key, api_key_env_var, base_url)

        if api_key is None:
            if "api.nvidia.com" in base_url:
                api_key = os.getenv("NVIDIA_API_KEY")
                if not api_key:
                    raise ValueError("NVIDIA_API_KEY is required for NVIDIA models and could not be found.")
            else:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key and "api.openai.com" in base_url:
                    raise ValueError("OPENAI_API_KEY is required for OpenAI models and could not be found.")
        return api_key

    def _is_reasoning_model(self, model_name: str) -> bool:
        if "gpt-5" in model_name:
            return True
        return re.match(r"^o\d", model_name)

    def _build_completion_request_params(self, **kwargs) -> dict:
        kwargs = copy.deepcopy(kwargs)
        assert kwargs.pop("tools", None) is None, "tools are not supported by completion requests."
        assert kwargs.pop("reasoning_effort", None) is None, (
            "reasoning_effort is not supported by completion requests."
        )
        assert kwargs.pop("top_k", -1) == -1, "`top_k` is not supported by OpenAI API, please set it to -1."
        assert kwargs.pop("min_p", 0.0) == 0.0, "`min_p` is not supported by OpenAI API, please set it to 0.0."
        assert kwargs.pop("repetition_penalty", 1.0) == 1.0, (
            "`repetition_penalty` is not supported by OpenAI API, please set it to 1.0."
        )
        if "tokens_to_generate" in kwargs:
            tokens_to_generate = kwargs.pop("tokens_to_generate")
            kwargs["max_tokens"] = tokens_to_generate
        if "random_seed" in kwargs:
            kwargs["seed"] = kwargs.pop("random_seed")
        if "stop_phrases" in kwargs:
            kwargs["stop"] = kwargs.pop("stop_phrases")
        return dict(kwargs)

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
    ) -> dict:
        # Validations
        if top_k != -1:
            raise ValueError("`top_k` is not supported by OpenAI API, please set it to -1.")
        if min_p > 0:
            raise ValueError("`min_p` is not supported by OpenAI API, please set it to 0.0.")
        if stream and top_logprobs is not None:
            raise ValueError("`top_logprobs` is not supported with stream=True.")
        if extra_body:
            raise ValueError("`extra_body` is not supported by OpenAI API")

        params = {
            "messages": messages,
            "seed": random_seed,
            "stop": stop_phrases or None,
            "timeout": timeout,
            "stream": stream,
            "tools": tools,
        }

        if self._is_reasoning_model(self.model):
            # Reasoning model specific validations and parameters
            if temperature != 0.0:
                raise ValueError(
                    "`temperature` is not supported by reasoning models, please set it to default value `0.0`."
                )
            if top_p != 0.95:
                raise ValueError(
                    "`top_p` is not supported by reasoning models, please set it to default value `0.95`."
                )
            if repetition_penalty != 1.0:
                raise ValueError(
                    "`repetition_penalty` is not supported by reasoning models, please set it to default value `1.0`."
                )
            if top_logprobs is not None:
                raise ValueError("`top_logprobs` is not supported by reasoning models, please set it to `None`.")

            params["max_completion_tokens"] = tokens_to_generate
            params["messages"] = [
                {**msg, "role": "developer"} if msg.get("role") == "system" else msg for msg in messages
            ]
            if reasoning_effort:
                params["reasoning_effort"] = reasoning_effort
                params["allowed_openai_params"] = ["reasoning_effort"]
        else:
            # Standard model parameters
            if reasoning_effort is not None:
                raise ValueError("`reasoning_effort` is only supported by reasoning models.")
            params["presence_penalty"] = repetition_penalty
            params["logprobs"] = top_logprobs is not None
            params["top_logprobs"] = top_logprobs
            params["max_completion_tokens"] = tokens_to_generate
            params["temperature"] = temperature
            params["top_p"] = top_p

        return params

    def _build_responses_request_params(self, input, **kwargs) -> dict:
        # Remapping variables to match responses API
        responses_params = self._build_chat_request_params(messages=input, **kwargs)
        responses_params["input"] = responses_params.pop("messages")
        responses_params["max_output_tokens"] = responses_params.pop("max_completion_tokens")
        return responses_params
