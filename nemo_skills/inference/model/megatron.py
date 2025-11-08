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

import openai

from .base import BaseModel


class MegatronModel(BaseModel):
    def __init__(self, **kwargs):
        # Megatron uses a non-standard base URL (no /v1) and a fixed model name.
        super().__init__(use_v1_endpoint=False, **kwargs)

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
        random_seed: int = None,
        stop_phrases: list[str] | None = None,
        timeout: int | None = None,
        top_logprobs: int | None = None,
        **kwargs,
    ) -> dict:
        # Validations
        if stream:
            raise NotImplementedError("Megatron server does not support streaming.")
        if min_p > 0:
            raise NotImplementedError("Megatron server does not support min_p parameter.")
        if repetition_penalty != 1.0:
            raise NotImplementedError("Megatron server does not support repetition_penalty parameter.")
        if top_k != -1:
            raise NotImplementedError("Megatron server does not support top_k parameter.")
        assert kwargs.get("tools") is None, "Megatron server does not support tools parameter."

        params = {
            "messages": messages,
            "max_tokens": tokens_to_generate,
            "temperature": temperature,
            "top_p": top_p,
            "seed": random_seed,
            "stop": stop_phrases or None,
            "logprobs": top_logprobs,
            "stream": stream,
            "echo": False,
            "n": 1,
            "logit_bias": None,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "timeout": timeout,
        }
        return params

    def _build_completion_request_params(
        self,
        prompt: str,
        stream: bool,
        tokens_to_generate: int = 512,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = -1,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = None,
        stop_phrases: list[str] | None = None,
        timeout: int | None = None,
        top_logprobs: int | None = None,
        **kwargs,
    ) -> dict:
        # Parameter validation specific to Megatron
        if stream:
            raise NotImplementedError("Megatron server does not support streaming.")
        if min_p > 0:
            raise NotImplementedError("Megatron server does not support min_p parameter.")
        if repetition_penalty != 1.0:
            raise NotImplementedError("Megatron server does not support repetition_penalty parameter.")
        if top_k != -1:
            raise NotImplementedError("Megatron server does not support top_k parameter.")
        assert kwargs.get("tools") is None, "Megatron server does not support tools parameter."

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
            "n": 1,
            "logit_bias": None,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "timeout": timeout,
        }

    def _parse_completion_response(
        self,
        response: "openai.types.Completion",
        include_response: bool = False,
        top_logprobs: int | None = None,
        **kwargs,
    ) -> dict | list[dict]:
        """Parse OpenAI response to extract the generated text and other metadata.

        Args:
            response: The response from OpenAI API

        Returns:
            A single dict with generation info
        """
        choice = response.choices[0]
        output = choice.text
        # adding back stop words - somehow sometimes it returns token ids, so we do not handle those for now
        if choice.finish_reason == "stop":
            if hasattr(choice, "stop_reason") and isinstance(choice.stop_reason, str):
                output += choice.stop_reason
            # sglang has a little different api here
            if hasattr(choice, "matched_stop") and isinstance(choice.matched_stop, str):
                output += choice.matched_stop

        result = {"generation": output, "num_generated_tokens": -1}
        if choice.logprobs and choice.logprobs.tokens:  # logprobs is always populated, but empty if not requested
            if top_logprobs is not None and top_logprobs != 0:
                result["logprobs"] = choice.logprobs.token_logprobs
                result["tokens"] = choice.logprobs.tokens
                result["top_logprobs"] = choice.logprobs.top_logprobs
            result["num_generated_tokens"] = len(choice.logprobs.tokens)
        if include_response:
            result["response"] = response
        return result

    def _parse_chat_completion_response(
        self,
        response: "openai.types.ChatCompletion",
        include_response: bool = False,
        top_logprobs: int | None = None,
        **kwargs,
    ) -> dict:
        choice = response.choices[0]
        output = choice.message.content
        if output is None:
            output = ""
        result = {"generation": output, "num_generated_tokens": response.usage.completion_tokens}

        if getattr(choice, "logprobs", None) and choice.logprobs.tokens:
            if top_logprobs is not None and top_logprobs != 0:
                result["logprobs"] = choice.logprobs.token_logprobs
                result["tokens"] = choice.logprobs.tokens
                result["top_logprobs"] = choice.logprobs.top_logprobs
            result["num_generated_tokens"] = len(choice.logprobs.tokens)
        if hasattr(choice, "finish_reason"):
            result["finish_reason"] = choice.finish_reason
        if include_response:
            result["response"] = response

        return result
