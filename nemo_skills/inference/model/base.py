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
import abc
import asyncio
import logging
import os
import uuid
from enum import Enum
from typing import Union

import httpx
import litellm
import litellm.constants
import litellm.llms.custom_httpx.http_handler
import litellm.llms.openai.common_utils
import openai

from nemo_skills.inference.patch_litellm_logging import patch_litellm_logging_worker
from nemo_skills.utils import get_logger_name

from .context_retry import ContextLimitRetryConfig, with_context_retry
from .utils import ServerTokenizer, WrapperAutoTokenizer, trim_after_stop_phrases

# litellm caches OpenAI/httpx clients with a 3600s (1hr) TTL. When the cached
# client expires it is garbage-collected, closing its httpx.AsyncClient and
# killing every in-flight request with "Cannot send a request, as the client
# has been closed". This is fine for short API calls but fatal for long-running
# generation jobs with high concurrency. The constant is copied into submodules
# via `from litellm.constants import ...`, so all three locations must be patched.
_EXTENDED_CLIENT_TTL = 14400
litellm.constants._DEFAULT_TTL_FOR_HTTPX_CLIENTS = _EXTENDED_CLIENT_TTL
litellm.llms.custom_httpx.http_handler._DEFAULT_TTL_FOR_HTTPX_CLIENTS = _EXTENDED_CLIENT_TTL
litellm.llms.openai.common_utils._DEFAULT_TTL_FOR_HTTPX_CLIENTS = _EXTENDED_CLIENT_TTL

LOG = logging.getLogger(get_logger_name(__file__))

# The logging worker sometimes does not stop. We patch it to disable its functionality.
# This issue is fixed in the latest litellm, keeping it here to avoid breaking previous containers
# We can remove it once everyone is moved to the latest container
patch_litellm_logging_worker()


class EndpointType(str, Enum):
    text = "text"
    chat = "chat"
    responses = "responses"


class BaseModel:
    """Base model class for handling requests to the inference server.

    Args:
        host: Optional[str] = '127.0.0.1' - Host of the inference server.
        port: Optional[str] = '5000' - Port of the inference server.
            Only required if handle_code_execution is True.
        ssh_server: Optional[str] = None - SSH server for tunneling requests.
            Useful if server is running on slurm cluster to which there is an ssh access
            Can also be specified through NEMO_SKILLS_SSH_SERVER env var.
        ssh_key_path: Optional[str] = None - Path to the ssh key for tunneling.
            Can also be specified through NEMO_SKILLS_SSH_KEY_PATH env var.
    """

    # Litellm provider name
    MODEL_PROVIDER = "openai"

    # Whether this model may use the native AsyncOpenAI + aiohttp fast-path
    # (see __init__). Only OpenAI-compatible endpoints that speak the plain
    # /v1/chat/completions schema at self.base_url can opt in. Subclasses that
    # talk to a non-OpenAI server (megatron), a differently-authed endpoint
    # (azure), a non-OpenAI provider (gemini), or need litellm's request
    # rewriting (multimodal audio) leave this False and stay on litellm.
    SUPPORTS_NATIVE_OPENAI = False

    # Keys that litellm understands but AsyncOpenAI's strict schema rejects as
    # unknown top-level fields; stripped before handing the body to the native
    # client. Keep this in sync with any litellm-only knob a _build_chat_request_params
    # implementation may emit.
    _LITELLM_ONLY_CHAT_PARAMS = frozenset({"allowed_openai_params"})

    def __init__(
        self,
        model: str,
        tokenizer: str | None = None,
        api_key: str | None = None,
        api_key_env_var: str | None = None,
        base_url: str | None = None,
        max_retries: int = 3,
        use_v1_endpoint: bool = True,
        host: str = "127.0.0.1",
        port: str = "5000",
        ssh_server: str | None = None,
        ssh_key_path: str | None = None,
        # Context limit retry config variables
        enable_soft_fail: bool = False,
        context_limit_retry_strategy: str | None = None,
        num_special_tokens_budget: int = 100,
        # Directory paths for data and output
        data_dir: str = "",
        output_dir: str | None = None,
        # Request tokenizer initialization independent of soft_fail
        require_tokenizer: bool = False,
        # Max in-flight HTTP requests gated by an internal semaphore.
        # The semaphore was originally added because earlier
        # combinations of httpx + litellm would hang under truly
        # unbounded concurrency. Modern httpx (>=0.27) handles this
        # fine, so the default is now high enough to effectively be
        # "no internal limit" -- the script-level concurrency knob
        # (generate.py `max_concurrent_requests`) is the single
        # user-facing cap. Override via $NEMO_SKILLS_MODEL_CONCURRENCY
        # only if you have a specific reason to throttle BELOW the
        # script-level cap.
        concurrent_requests_limit: int = int(os.environ.get("NEMO_SKILLS_MODEL_CONCURRENCY", "65536")),
    ):
        self._tunnel = None
        self.model_name_or_path = model
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.server_host = host
        self.server_port = port
        self.ssh_server = ssh_server
        self.ssh_key_path = ssh_key_path
        self.context_limit_retry_config = ContextLimitRetryConfig(
            enable_soft_fail=enable_soft_fail,
            strategy=context_limit_retry_strategy,
            num_special_tokens_budget=num_special_tokens_budget,
        )
        if ssh_server is None:
            self.ssh_server = os.getenv("NEMO_SKILLS_SSH_SERVER")
        if ssh_key_path is None:
            self.ssh_key_path = os.getenv("NEMO_SKILLS_SSH_KEY_PATH")

        if self.ssh_server and self.ssh_key_path:
            import sshtunnel

            if "@" in self.ssh_server:
                ssh_username, ssh_server = self.ssh_server.split("@")
            else:
                ssh_server = self.ssh_server
                ssh_username = None

            self._tunnel = sshtunnel.SSHTunnelForwarder(
                (ssh_server, 22),
                ssh_username=ssh_username,
                ssh_pkey=self.ssh_key_path,
                remote_bind_address=(self.server_host, int(self.server_port)),
            )
            self._tunnel.start()
            self.server_host = "127.0.0.1"
            self.server_port = str(self._tunnel.local_bind_port)

        if base_url is None:
            v1_suffix = "/v1" if use_v1_endpoint else ""
            self.base_url = f"http://{self.server_host}:{self.server_port}{v1_suffix}"
        elif base_url == "":
            # We don't want to use base_url if it is an empty string
            self.base_url = None
        else:
            self.base_url = base_url

        needs_retry_tokenizer = enable_soft_fail and context_limit_retry_strategy in {
            "reduce_prompt_from_start",
            "reduce_prompt_from_end",
        }
        if require_tokenizer or needs_retry_tokenizer:
            self.tokenizer = self._get_tokenizer(tokenizer)
        else:
            self.tokenizer = None

        api_key = self._get_api_key(api_key, api_key_env_var, base_url)
        if api_key is None:  # self-hosted models don't need the key, but still require the parameter
            api_key = "EMPTY"

        model_litellm = f"{self.MODEL_PROVIDER}/{model}"
        # Passed to litellm every time we call it
        self.litellm_kwargs = dict(
            model=model_litellm,
            max_retries=max_retries,
            api_key=api_key,
            base_url=self.base_url,
            api_base=self.base_url,  # Used in later versions with responses API
        )
        httpx_limits = httpx.Limits(max_keepalive_connections=None, max_connections=None)
        litellm.client_session = httpx.Client(limits=httpx_limits)
        litellm.aclient_session = httpx.AsyncClient(limits=httpx_limits)
        # Controlling concurrent requests using semaphore since large
        # concurrent requests result into httpx hanging
        self.concurrent_semaphore = asyncio.Semaphore(concurrent_requests_limit)

        # Native aiohttp fast-path (DEFAULT for OpenAI-compatible
        # providers). The litellm.acompletion path goes through httpx,
        # a pure-Python HTTP implementation -- at 5k+ concurrent
        # connections it becomes the GIL-bound bottleneck (~95% of
        # single-core CPU in the async loop, per py-spy). The OpenAI
        # SDK natively supports an aiohttp transport via
        # DefaultAioHttpClient. aiohttp's request/response parsing is
        # C-extension backed, so the same workload runs at <5% CPU on
        # the same hardware.
        #
        # Enabled by default for providers with SUPPORTS_NATIVE_OPENAI
        # (openai, vllm, sglang). Both the non-streaming and streaming
        # chat endpoints go through this path; text completions, the
        # responses API, and non-OpenAI-shaped providers (gemini,
        # azure, megatron, multimodal) still use litellm. Set
        # NEMO_SKILLS_OPENAI_AIOHTTP=0 to force everything back onto
        # litellm/httpx.
        self._async_openai_client = None
        native_enabled = os.environ.get("NEMO_SKILLS_OPENAI_AIOHTTP", "1") != "0"
        if native_enabled and self.SUPPORTS_NATIVE_OPENAI:
            try:
                from openai import AsyncOpenAI, DefaultAioHttpClient

                # openai SDK defaults the aiohttp connector to
                # max_connections=1000, max_keepalive_connections=100,
                # which silently caps the lane count at high
                # concurrency (8k lanes see only ~1000 open TCP
                # connections; the rest stall in the aiohttp connector
                # queue). Also, httpx-aiohttp's transport translates
                # httpx.Limits(max_connections=None) to "omit the
                # `limit` kwarg," which makes aiohttp fall back to ITS
                # default of 100 -- worse than the openai SDK
                # default. We therefore pass an explicit large
                # integer (NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT, default
                # 65536) instead of None.
                aiohttp_limit = int(os.environ.get("NEMO_SKILLS_OPENAI_AIOHTTP_LIMIT", "65536"))
                self._async_openai_client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=self.base_url,
                    http_client=DefaultAioHttpClient(
                        limits=httpx.Limits(
                            max_connections=aiohttp_limit,
                            max_keepalive_connections=aiohttp_limit,
                        ),
                    ),
                    # Mirror the litellm path's retry budget (self.litellm_kwargs
                    # also passes max_retries). The SDK's own retry loop honors
                    # Retry-After / backoff for HTTP 429 and 5xx -- error classes
                    # that our app-level transient-retry except clause does NOT
                    # catch (it only handles connection-class errors). Leaving this
                    # at 0 silently dropped 429/5xx under load on the native path,
                    # which is the DEFAULT for openai/vllm/sglang. SDK retries run
                    # inside our concurrent_semaphore hold, so one logical request
                    # still counts as one in-flight slot.
                    max_retries=max_retries,
                )
            except Exception as e:
                # ImportError if openai[aiohttp] isn't installed, but the
                # DefaultAioHttpClient constructor can also raise other errors
                # when the aiohttp extra is only partially present. Degrade to
                # litellm/httpx rather than failing model construction.
                LOG.warning(
                    "Native AsyncOpenAI aiohttp fast-path unavailable (%s: %s); "
                    "falling back to litellm/httpx. Install openai[aiohttp] to enable it.",
                    type(e).__name__,
                    e,
                )
                self._async_openai_client = None

    def _get_api_key(self, api_key: str | None, api_key_env_var: str | None, base_url: str) -> str | None:
        if api_key:  # explicit cmd argument always takes precedence
            return api_key
        if api_key_env_var:
            api_key = os.getenv(api_key_env_var)
            if not api_key:
                raise ValueError(
                    f"You defined api_key_env_var={api_key_env_var} but the value is not set. "
                    f"Either remove api_key_env_var or set {api_key_env_var}=<some value>. "
                    "Did you forget to add it to your cluster config?"
                )
        return api_key

    def __del__(self):
        # getattr guard: __del__ can fire on an instance whose __init__ raised
        # (or never ran), before _tunnel was set.
        tunnel = getattr(self, "_tunnel", None)
        if tunnel:
            tunnel.stop()

    def _maybe_apply_stop_phrase_removal(
        self, result: dict, remove_stop_phrases: bool, stop_phrases: list[str] | None
    ) -> None:
        if remove_stop_phrases:
            result["generation"] = trim_after_stop_phrases(result["generation"], stop_phrases)

    def _get_tokenizer(self, tokenizer: str | None) -> Union[ServerTokenizer, WrapperAutoTokenizer, None]:
        """Initialize the tokenizer from the string, otherwise initialize the tokenizer endpoint"""
        # Try to initialize the tokenizer from tokenizer string
        for tokenizer_string in [tokenizer, self.model_name_or_path]:
            if tokenizer_string is None:
                continue

            wrapped_tokenizer = self._initialize_tokenizer(tokenizer_string)
            if wrapped_tokenizer is not None:
                return wrapped_tokenizer

        # Try to initialize the tokenizer endpoint
        tokenizer_endpoint = self._get_tokenizer_endpoint()
        if tokenizer_endpoint is not None:
            return tokenizer_endpoint

        # No tokenizer found
        LOG.info(f"No tokenizer found for model: {self.model_name_or_path}")
        return None

    def _get_tokenizer_endpoint(self) -> str | None:
        """Get the tokenizer endpoint if available."""
        return None

    def _initialize_tokenizer(self, tokenizer: str | None) -> WrapperAutoTokenizer | None:
        if tokenizer is None:
            return None
        if isinstance(tokenizer, str):
            try:
                return WrapperAutoTokenizer(tokenizer)
            except OSError:
                LOG.warning(f"Tokenizer not found at '{tokenizer}', trying fallback to server /tokenize endpoint")
                return None

    @abc.abstractmethod
    def _build_chat_request_params(self, **kwargs) -> dict:
        pass

    @abc.abstractmethod
    def _build_completion_request_params(self, **kwargs) -> dict:
        pass

    def _build_responses_request_params(self, **kwargs) -> dict:
        raise NotImplementedError("Responses completion is not not supported or implemented for this model.")

    @staticmethod
    def _apply_routing_keys(request_params: dict, cache_key: str | None) -> None:
        """Attach optional, provider-agnostic request-routing keys. Plain
        OpenAI/vLLM/gemini endpoints ignore (or echo) them.

          * `X-Request-Id` (extra_headers) = per-request id for idempotency /
            tracing. Kept stable across the OpenAI SDK's internal retry loop so
            a retry is identifiable as the same logical call; distinct calls
            (incl. parallel samples) get distinct ids.
          * `prompt_cache_key` (opt-in via `cache_key`) = the standard
            OpenAI/vLLM prefix-cache hint. Requests that share a value are
            hinting they share a common prompt prefix and can reuse a warm
            prefix cache. Omitted when cache_key is None.
        """
        request_params.setdefault("extra_headers", {}).setdefault("X-Request-Id", str(uuid.uuid4()))
        if cache_key is not None:
            request_params["prompt_cache_key"] = cache_key

    @with_context_retry
    async def generate_async(
        self,
        prompt: str | list[dict],
        endpoint_type: EndpointType = None,
        tokens_to_generate: int | None = None,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = -1,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = None,
        stop_phrases: list[str] | None = None,
        top_logprobs: int | None = None,
        timeout: float | int | None = 14400,  # None is 10min
        remove_stop_phrases: bool = True,
        stream: bool = False,
        reasoning_effort: str | None = None,
        tools: list[dict] | None = None,
        include_response: bool = False,
        extra_body: dict = None,
        response_format=None,
        cache_key: str | None = None,
    ) -> dict:
        if endpoint_type is None:
            # Infering completion type from prompt
            endpoint_type = EndpointType.chat if isinstance(prompt, list) else EndpointType.text
        # Check tool calls are a list of dict
        if tools is not None:
            for tool in tools:
                # TODO: We may want to add additional checks for tools in the future
                if not isinstance(tool, dict):
                    raise ValueError(f"Tool must be a dictionary, got {type(tool)}")

        kwargs = {
            "tokens_to_generate": tokens_to_generate,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "random_seed": random_seed,
            "stop_phrases": stop_phrases,
            "top_logprobs": top_logprobs,
            "timeout": timeout,
            "reasoning_effort": reasoning_effort,
            "tools": tools,
            "extra_body": extra_body,
            "response_format": response_format,
        }

        # Two separate retry budgets:
        # - bad_request_count is for the gpt-oss "output messages"
        #   bug; small, app-level. Kept as-is until we drop that
        #   model (or vLLM fixes it upstream).
        # - transient_count handles socket-level failures that the
        #   raw HTTP path can hit under burst load (kernel SYN-drops
        #   when the accept queue overflows, brief upstream
        #   unavailability during worker turnover, etc). litellm's
        #   own max_retries covers HTTP 429/5xx but NOT these
        #   connection-class exceptions, so a single ramp-up burst
        #   into the upstream could otherwise hard-fail individual
        #   datapoints. Retries here turn those into invisible
        #   ~50-200ms re-tries.
        max_retries = 2
        retry_count = 0
        max_transient_retries = int(os.environ.get("NEMO_SKILLS_TRANSIENT_RETRIES", "3"))
        transient_count = 0

        async with self.concurrent_semaphore:
            while retry_count <= max_retries:
                try:
                    if endpoint_type == EndpointType.chat:
                        assert isinstance(prompt, list), "Chat completion requests must be a list of messages."
                        request_params = self._build_chat_request_params(messages=prompt, stream=stream, **kwargs)
                        self._apply_routing_keys(request_params, cache_key)
                        # Fast path: native AsyncOpenAI + aiohttp transport,
                        # set up in __init__ for OpenAI-compatible providers
                        # (disable with NEMO_SKILLS_OPENAI_AIOHTTP=0). Skips
                        # litellm's httpx-based dispatch which is the dominant
                        # single-core CPU consumer at high concurrency. Both
                        # streaming and non-streaming chat are routed here; the
                        # responses API, text completions, and non-OpenAI
                        # providers still fall through to litellm.
                        use_native = self._async_openai_client is not None
                        if use_native:
                            # Strip (a) litellm-only knobs the strict SDK rejects
                            # and (b) None-valued params. litellm OMITS None-valued
                            # fields from the wire body, but AsyncOpenAI serializes
                            # an explicit None as JSON `null` -- e.g. the vllm builder
                            # emits response_format/tools/stop/seed=None, and some
                            # OpenAI-compatible servers reject `"response_format": null`.
                            # Dropping None here keeps the native body byte-compatible
                            # with what the litellm path sent.
                            native_params = {
                                k: v
                                for k, v in request_params.items()
                                if k not in self._LITELLM_ONLY_CHAT_PARAMS and v is not None
                            }
                            # AsyncOpenAI needs `model` as a top-level
                            # kwarg; litellm consumes it via
                            # self.litellm_kwargs (with the `openai/`
                            # provider prefix). Use the raw model name
                            # here.
                            native_params.setdefault("model", self.model_name_or_path)
                            response = await self._async_openai_client.chat.completions.create(
                                **native_params,
                            )
                        else:
                            response = await litellm.acompletion(**request_params, **self.litellm_kwargs)
                        if stream:
                            result = self._stream_chat_chunks_async(response)
                        else:
                            result = self._parse_chat_completion_response(
                                response, include_response=include_response, **kwargs
                            )
                    elif endpoint_type == EndpointType.text:
                        assert isinstance(prompt, str), "Text completion requests must be a string."
                        request_params = self._build_completion_request_params(prompt=prompt, stream=stream, **kwargs)
                        self._apply_routing_keys(request_params, cache_key)
                        response = await litellm.atext_completion(**request_params, **self.litellm_kwargs)
                        if stream:
                            result = self._stream_completion_chunks_async(response)
                        else:
                            result = self._parse_completion_response(
                                response, include_response=include_response, **kwargs
                            )
                    elif endpoint_type == EndpointType.responses:
                        assert isinstance(prompt, list), "Responses completion requests must be a list."
                        request_params = self._build_responses_request_params(input=prompt, stream=stream, **kwargs)
                        self._apply_routing_keys(request_params, cache_key)
                        response = await litellm.aresponses(**request_params, **self.litellm_kwargs)
                        if stream:
                            raise NotImplementedError("Streaming responses is not supported yet.")
                        else:
                            result = self._parse_responses_completion_response(
                                response, include_response=include_response, **kwargs
                            )
                    else:
                        raise TypeError(f"Unsupported completion type: {endpoint_type}")
                    if not stream:
                        self._maybe_apply_stop_phrase_removal(result, remove_stop_phrases, stop_phrases)
                    return result

                except openai.BadRequestError as e:
                    if "output messages (reasoning and final)" in str(e):
                        if retry_count < max_retries:
                            retry_count += 1
                            LOG.warning(f"BadRequestError, retrying {retry_count}/{max_retries}: {e}")
                            continue

                        LOG.error(f"BadRequestError after {max_retries} retries, returning empty response: {e}")
                        return {
                            "generation": "",
                            "reasoning_content": "",
                            "num_generated_tokens": 0,
                            "serialized_output": [],
                        }
                    else:
                        raise e
                except (
                    httpx.ConnectError,
                    httpx.ReadError,
                    httpx.RemoteProtocolError,
                    httpx.PoolTimeout,
                    httpx.NetworkError,
                    openai.APIConnectionError,
                    ConnectionError,
                    ConnectionResetError,
                ) as e:
                    # Transient socket-level failure. Common cause at
                    # high concurrency: kernel SYN-drop during ramp-up
                    # when the upstream accept queue briefly overflows.
                    # Backoff doubles each retry (50ms, 100ms, 200ms)
                    # to let the accept queue drain.
                    if transient_count < max_transient_retries:
                        transient_count += 1
                        backoff = 0.05 * (2 ** (transient_count - 1))
                        LOG.warning(
                            "transient connect error, retry %d/%d after %.0fms: %s: %s",
                            transient_count,
                            max_transient_retries,
                            backoff * 1000,
                            type(e).__name__,
                            e,
                        )
                        await asyncio.sleep(backoff)
                        continue
                    LOG.error(
                        "transient connect error after %d retries, propagating: %s: %s",
                        max_transient_retries,
                        type(e).__name__,
                        e,
                    )
                    raise

        return result

    def _parse_completion_response(
        self, response: "openai.types.Completion", include_response: bool = False, **kwargs
    ) -> dict:
        choice = response.choices[0]
        output = choice.text
        if output is None:
            output = ""

        # In some cases, the stop reason is not included in the text, so we add it back
        if choice.finish_reason == "stop":
            if hasattr(choice, "stop_reason") and isinstance(choice.stop_reason, str):
                output += choice.stop_reason
            # sglang has a little different api here
            if hasattr(choice, "matched_stop") and isinstance(choice.matched_stop, str):
                output += choice.matched_stop

        result = {"generation": output, "num_generated_tokens": response.usage.completion_tokens}
        if getattr(response.usage, "prompt_tokens", None) is not None:
            result["num_input_tokens"] = response.usage.prompt_tokens
        elif getattr(response.usage, "input_tokens", None) is not None:
            result["num_input_tokens"] = response.usage.input_tokens
        if getattr(choice, "logprobs", None):
            result["logprobs"] = choice.logprobs.token_logprobs
            result["tokens"] = choice.logprobs.tokens
            result["top_logprobs"] = choice.logprobs.top_logprobs
        if choice.finish_reason:
            result["finish_reason"] = choice.finish_reason

        if include_response:
            result["response"] = response

        return result

    @staticmethod
    def _extract_reasoning(message_or_delta) -> str | None:
        """Return reasoning text from an OpenAI chat message or streaming delta,
        tolerating both field names.

        We normalize the field name here manually because litellm typically
        takes care of it for us (it maps vLLM's field to `reasoning_content`),
        but the native AsyncOpenAI fast-path bypasses litellm and sees vLLM's
        raw field. vLLM historically exposed it as `reasoning_content`; newer
        versions use `reasoning`. We accept either, preferring `reasoning_content`
        when both are present.
        """
        for attr in ("reasoning_content", "reasoning"):
            value = getattr(message_or_delta, attr, None)
            if value:
                return value
        return None

    def _parse_chat_completion_response(self, response, include_response: bool = False, **kwargs) -> dict:
        choice = response.choices[0]
        output = choice.message.content
        if output is None:
            output = ""
        result = {"generation": output, "num_generated_tokens": response.usage.completion_tokens}
        if getattr(response.usage, "prompt_tokens", None) is not None:
            result["num_input_tokens"] = response.usage.prompt_tokens
        elif getattr(response.usage, "input_tokens", None) is not None:
            result["num_input_tokens"] = response.usage.input_tokens

        # Add reasoning_content if available (accepts .reasoning or .reasoning_content)
        reasoning = self._extract_reasoning(choice.message)
        if reasoning:
            result["reasoning_content"] = reasoning

        # Dump the full usage object verbatim into the output. Keeping
        # None-valued fields (rather than excluding them) preserves the
        # distinction between "server explicitly sent null" and "field
        # was absent" — both can be meaningful (e.g. some providers
        # send prompt_tokens_details=null when there's no cache hit).
        try:
            result["usage"] = response.usage.model_dump()
        except AttributeError:
            # Older or non-pydantic usage objects: fall back to dict().
            try:
                result["usage"] = dict(response.usage)
            except Exception:
                result["usage"] = {
                    "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(response.usage, "completion_tokens", None),
                    "total_tokens": getattr(response.usage, "total_tokens", None),
                }

        # Extract detailed token breakdown for reasoning models if available
        if hasattr(response.usage, "completion_tokens_details") and response.usage.completion_tokens_details:
            details = response.usage.completion_tokens_details
            if hasattr(details, "reasoning_tokens") and details.reasoning_tokens is not None:
                result["num_reasoning_tokens"] = details.reasoning_tokens
                result["num_answer_tokens"] = response.usage.completion_tokens - details.reasoning_tokens

        if getattr(choice, "logprobs", None) and choice.logprobs.content:
            result["logprobs"] = [tok.logprob for tok in choice.logprobs.content]
            result["tokens"] = [tok.token for tok in choice.logprobs.content]
            result["top_logprobs"] = []
            for token_logprob in choice.logprobs.content:
                logprob = {entry.token: entry.logprob for entry in token_logprob.top_logprobs}
                if token_logprob.token not in logprob:
                    logprob[token_logprob.token] = token_logprob.logprob
                result["top_logprobs"].append(logprob)
        if choice.finish_reason:
            result["finish_reason"] = choice.finish_reason
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            result["tool_calls"] = choice.message.tool_calls
        result["serialized_output"] = self._serialize_output(response)
        if include_response:
            result["response"] = response

        return result

    def _process_completion_chunk(self, chunk, emitted_so_far: list):
        """Process a single completion chunk and return data to yield."""
        cur_delta = chunk.choices[0].text
        emitted_so_far.append(cur_delta)

        results_to_yield = []
        if cur_delta:
            results_to_yield.append({"generation": cur_delta})

        # vllm variant
        stop_reason = getattr(chunk.choices[0], "stop_reason", None)
        # sglang variant
        matched_stop = getattr(chunk.choices[0], "matched_stop", None)

        # vllm variant - emit stop_reason as is and finish
        if stop_reason and isinstance(stop_reason, str):
            results_to_yield.append({"generation": stop_reason})

        # sglang variant - emit only not-yet-sent part of matched_stop
        if matched_stop and isinstance(matched_stop, str):
            remaining = matched_stop
            # find the longest prefix of matched_stop that is already at
            # the end of what we've emitted.
            emitted_str = "".join(emitted_so_far)
            max_len = min(len(emitted_str), len(matched_stop))
            for i in range(max_len, 0, -1):
                if emitted_str.endswith(matched_stop[:i]):
                    remaining = matched_stop[i:]
                    break
            if remaining:
                results_to_yield.append({"generation": remaining})

        return results_to_yield

    def _process_chat_chunk(self, chunk):
        """Process a single chat chunk and return data to yield."""
        if hasattr(chunk.choices[0], "delta"):
            cur_delta = chunk.choices[0].delta.content
            # Check for reasoning in delta (accepts .reasoning or .reasoning_content)
            reasoning_delta = self._extract_reasoning(chunk.choices[0].delta)
            tool_calls_delta = getattr(chunk.choices[0].delta, "tool_calls", None)
        else:
            cur_delta = chunk.choices[0].text
            reasoning_delta = None
            tool_calls_delta = None

        finish_reason = getattr(chunk.choices[0], "finish_reason", None)
        result = {"generation": cur_delta or ""}

        # Add reasoning_content to result if available
        if reasoning_delta:
            result["reasoning_content"] = reasoning_delta

        if tool_calls_delta:
            result["tool_calls"] = tool_calls_delta

        if finish_reason:
            result["finish_reason"] = finish_reason
            if not cur_delta:
                result["generation"] = ""

        return [result]

    async def _stream_completion_chunks_async(self, response):
        emitted_so_far = []
        async for chunk in response:
            results = self._process_completion_chunk(chunk, emitted_so_far)
            for result in results:
                yield result

    def _parse_responses_completion_response(self, response, include_response: bool = False, **kwargs) -> dict:
        """Public method for parsing responses API responses"""
        result = {"generation": "", "num_generated_tokens": 0}

        if hasattr(response, "usage"):
            result["num_generated_tokens"] = response.usage.output_tokens

        tool_calls = []
        reasoning_content = ""
        generation_text = ""

        if hasattr(response, "output") and response.output:
            for output_item in response.output:
                # Handle reasoning content
                if output_item.type == "reasoning":
                    if output_item.content:
                        for content_item in output_item.content:
                            if content_item.text:
                                reasoning_content += content_item.text + "\n"

                # Handle function calls
                elif output_item.type == "function_call":
                    tool_calls.append(output_item)

                # Handle message content
                elif output_item.type == "message":
                    if output_item.content:
                        for content_item in output_item.content:
                            if content_item.text:
                                generation_text += content_item.text

        if tool_calls:
            result["tool_calls"] = tool_calls
            result["generation"] = ""  # No text generation when there are tool calls
        else:
            result["generation"] = generation_text
        if reasoning_content:
            result["reasoning_content"] = reasoning_content.strip()

        result["finish_reason"] = response.status
        result["serialized_output"] = self._serialize_output(response)
        if include_response:
            result["response"] = response

        return result

    def _serialize_output(self, response):
        """Serialize response output objects using model_dump() for conversation history."""
        serialized_output = []

        if hasattr(response, "output") and response.output:
            for output_item in response.output:
                serialized_output.append(output_item.model_dump())
        elif hasattr(response, "choices") and response.choices:
            for choice in response.choices:
                serialized_output.append(choice.model_dump()["message"])
        else:
            raise ValueError(f"Unsupported response type: {type(response)}")
        return serialized_output

    async def _stream_chat_chunks_async(self, response):
        async for chunk in response:
            # Some servers emit chunks with no choices (e.g. a trailing
            # usage-only chunk when stream_options.include_usage is set, or
            # keepalive chunks). They carry no delta -- skip them so
            # _process_chat_chunk's choices[0] access stays safe.
            if not getattr(chunk, "choices", None):
                continue
            results = self._process_chat_chunk(chunk)
            for result in results:
                yield result
