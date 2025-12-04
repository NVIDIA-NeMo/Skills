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

from .vllm import VLLMModel


class SGLangModel(VLLMModel):
    """SGLang model that extends VLLMModel with proper tool_choice handling.

    SGLang requires "tool_choice": "auto" in the request body when tools are provided,
    unlike VLLM which uses a server argument (--enable-auto-tool-choice).
    """

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
        request = super()._build_chat_request_params(
            messages=messages,
            stream=stream,
            tokens_to_generate=tokens_to_generate,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            random_seed=random_seed,
            stop_phrases=stop_phrases,
            timeout=timeout,
            top_logprobs=top_logprobs,
            reasoning_effort=reasoning_effort,
            tools=tools,
            extra_body=extra_body,
        )
        # SGLang requires tool_choice in the request body when tools are provided
        if tools is not None:
            request["tool_choice"] = "auto"
        return request
