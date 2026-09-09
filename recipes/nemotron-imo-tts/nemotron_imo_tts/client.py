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

"""The request layer: one OpenAI-compatible client per endpoint, per-role lanes, context budget, retry policy.

The pipeline owns retries (the SDK's are disabled). Transient failures (408, 429, 5xx, timeouts, connection
errors) are retried with jittered exponential backoff until ``retry_window_s`` elapses, then the request is
parked. Deterministic failures (400 and friends, context-length errors) are parked immediately. Authentication
and missing-model errors (401, 403, 404) are fatal for the run. A parked request is a contained failure: the
caller records it and continues with one fewer sample, vote, or judgment.
"""

import asyncio
import errno
import json
import logging
import random
import threading
import time
import uuid
from dataclasses import dataclass, field

import httpx
import openai

from nemotron_imo_tts.config import resolve_api_key
from nemotron_imo_tts.text import sha256_text

LOG = logging.getLogger("nemotron_imo_tts.client")

# One stable request id per logical unit of work, constant across attempts, so idempotent gateways can reattach.
REQUEST_ID_NAMESPACE = uuid.UUID("2d6f6f1e-4cbb-5d0e-9a4c-9f1e0c3a7b21")

PARK_STATUSES = frozenset({400, 409, 413, 415, 422})
FATAL_STATUSES = frozenset({401, 403, 404})
RETRY_STATUSES = frozenset({408, 429})
CONTEXT_MARKERS = (
    "maximum context length",
    "context length exceeded",
    "context window",
    "too many tokens",
    "prompt is too long",
    "input is too long",
)
ROLE_LANES = {"gen": "generation", "verify": "verification", "refine": "refinement", "judge": "judging"}
TOKENIZER_MAX_CONCURRENT = 4


@dataclass
class Response:
    generation: str
    finish_reason: str | None
    reasoning_content: str | None
    usage: dict | None
    num_generated_tokens: int | None
    attempts: int
    elapsed_s: float
    context_budget: dict = field(default_factory=dict)
    replayed: bool = False


class RequestParked(Exception):
    """A contained per-request failure. ``details`` is JSON-serializable and is persisted by the caller."""

    def __init__(self, details):
        self.details = dict(details)
        super().__init__(self.details.get("message") or self.details.get("category") or "request parked")


class FatalRequestError(Exception):
    """A failure that can never be per-request (bad credentials, unknown model or route, file descriptors)."""

    def __init__(self, details):
        self.details = dict(details)
        super().__init__(self.details.get("message") or "fatal request error")


def status_code(exc):
    status = getattr(exc, "status_code", None)
    if status is None and getattr(exc, "response", None) is not None:
        status = getattr(exc.response, "status_code", None)
    return status


def classify_error(exc):
    """Map one request failure to ``"fatal"``, ``"park"``, or ``"retry"``."""
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in (errno.EMFILE, errno.ENFILE):
        return "fatal"
    status = status_code(exc)
    if status in FATAL_STATUSES:
        return "fatal"
    message = str(exc).lower()
    if status in PARK_STATUSES or any(marker in message for marker in CONTEXT_MARKERS):
        return "park"
    if isinstance(exc, (openai.APITimeoutError, openai.APIConnectionError)):
        return "retry"
    if status in RETRY_STATUSES or (status is not None and status >= 500):
        return "retry"
    return "park"


def error_details(exc, attempts, budget):
    return {
        "exception_class": type(exc).__name__,
        "status_code": status_code(exc),
        "message": str(exc)[:2000],
        "attempts": attempts,
        "context_budget": dict(budget),
    }


class ContextBudgeter:
    """Fit the completion budget to the model context using the served model's tokenizer (optional)."""

    def __init__(self, tokenizer_name_or_path, safety_margin_tokens):
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.safety_margin_tokens = int(safety_margin_tokens)
        self._tokenizer = None
        self._lock = threading.Lock()
        self._cache = {}
        self._semaphore = asyncio.Semaphore(TOKENIZER_MAX_CONCURRENT)

    def load(self):
        """Load the tokenizer now; called at preflight so a missing or broken tokenizer fails before any request.

        The lock also keeps concurrent lazy loads from importing transformers in two threads at once.
        """
        with self._lock:
            if self._tokenizer is None and self.tokenizer_name_or_path:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name_or_path)
        return self._tokenizer

    async def count_tokens(self, messages):
        key = sha256_text(json.dumps(messages, sort_keys=True, ensure_ascii=False))
        if key in self._cache:
            return self._cache[key]
        async with self._semaphore:
            tokenizer = await asyncio.to_thread(self.load)
            if tokenizer is None:
                return None

            def count():
                ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
                if not isinstance(ids, list):
                    ids = ids["input_ids"]
                return len(ids)

            prompt_tokens = await asyncio.to_thread(count)
        self._cache[key] = prompt_tokens
        return prompt_tokens

    async def budget(self, messages, spec):
        context = {
            "prompt_tokens": None,
            "requested_max_tokens": int(spec.max_tokens),
            "effective_max_tokens": int(spec.max_tokens),
            "model_max_len": int(spec.max_len),
            "safety_margin_tokens": self.safety_margin_tokens,
        }
        if not self.tokenizer_name_or_path and self._tokenizer is None:
            return context
        prompt_tokens = await self.count_tokens(messages)
        if prompt_tokens is None:
            return context
        available = int(spec.max_len) - int(prompt_tokens) - self.safety_margin_tokens
        context["prompt_tokens"] = int(prompt_tokens)
        context["effective_max_tokens"] = min(int(spec.max_tokens), max(0, available))
        if available <= 0:
            context["error_type"] = "prompt_too_long"
        return context


class RequestLayer:
    def __init__(self, cfg, budgeter, *, transport=None, clock=time.monotonic, sleep=asyncio.sleep):
        self.cfg = cfg
        self.budgeter = budgeter
        self._transport = transport
        self._clock = clock
        self._sleep = sleep
        self._clients = {}
        self._lanes = {role: asyncio.Semaphore(getattr(cfg.concurrency, lane)) for role, lane in ROLE_LANES.items()}

    def _client(self, spec):
        key = (spec.base_url, spec.api_key_env)
        if key not in self._clients:
            http_client = None
            if self._transport is not None:
                http_client = httpx.AsyncClient(transport=self._transport, timeout=self.cfg.endpoint.timeout_s)
            self._clients[key] = openai.AsyncOpenAI(
                base_url=spec.base_url,
                api_key=resolve_api_key(spec),
                timeout=self.cfg.endpoint.timeout_s,
                max_retries=0,
                http_client=http_client,
            )
        return self._clients[key]

    async def aclose(self):
        for client in self._clients.values():
            await client.close()
        self._clients.clear()

    async def list_models(self, spec):
        """Model ids served by the endpoint, or None when the listing is unavailable."""
        try:
            page = await self._client(spec).models.list()
            return [item.id for item in page.data]
        except Exception as exc:  # the listing is informational only
            LOG.debug("model listing unavailable for %s: %s", spec.base_url, exc)
            return None

    def _backoff(self, attempts, exc):
        base = max(0.0, float(self.cfg.endpoint.backoff_base_s))
        cap = max(0.0, float(self.cfg.endpoint.backoff_max_s))
        delay = min(cap, base * (2 ** max(0, attempts - 1)))
        response = getattr(exc, "response", None)
        retry_after = getattr(response, "headers", {}).get("retry-after") if response is not None else None
        if retry_after is not None:
            try:
                return min(cap, max(0.0, float(retry_after)))
            except ValueError:
                pass
        return random.uniform(0.0, delay) if delay > 0 else 0.0

    async def request(self, role, spec, messages, seed, identity):
        """Send one logical request. Raises RequestParked, FatalRequestError, or asyncio.CancelledError."""
        budget = await self.budgeter.budget(messages, spec)
        if budget.get("error_type") == "prompt_too_long":
            raise RequestParked(
                {
                    "category": "prompt_too_long",
                    "status_code": None,
                    "message": "Prompt plus safety margin leaves no completion budget",
                    "attempts": 0,
                    "context_budget": dict(budget),
                }
            )
        request_id = str(uuid.uuid5(REQUEST_ID_NAMESPACE, identity))
        started = self._clock()
        attempts = 0
        last_exc = None
        lane = self._lanes[role]
        while True:
            attempts += 1
            completion = None
            async with lane:
                task = asyncio.create_task(
                    self._client(spec).chat.completions.create(
                        model=spec.model,
                        messages=messages,
                        max_tokens=budget["effective_max_tokens"],
                        temperature=self.cfg.sampling.temperature,
                        top_p=self.cfg.sampling.top_p,
                        seed=seed,
                        extra_headers={"X-Request-Id": request_id},
                        extra_body=self.cfg.endpoint.extra_body or None,
                    )
                )
                try:
                    completion = await asyncio.shield(task)
                except asyncio.CancelledError:
                    # A call that finished in the same scheduling turn wins over the cancellation.
                    if task.done() and not task.cancelled() and task.exception() is None:
                        completion = task.result()
                    else:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                        raise
                except Exception as exc:
                    last_exc = exc
                    kind = classify_error(exc)
                    if kind == "fatal":
                        raise FatalRequestError({**error_details(exc, attempts, budget), "category": "fatal"}) from exc
                    if kind == "park":
                        raise RequestParked(
                            {**error_details(exc, attempts, budget), "category": "parked_deterministic"}
                        ) from exc
            if completion is not None:
                return self._normalize(completion, attempts, self._clock() - started, budget)
            if self._clock() - started >= self.cfg.endpoint.retry_window_s:
                raise RequestParked(
                    {**error_details(last_exc, attempts, budget), "category": "retry_exhausted"}
                ) from last_exc
            LOG.warning(
                "transient failure on %s (attempt %d): %s: %s", identity, attempts, type(last_exc).__name__, last_exc
            )
            await self._sleep(self._backoff(attempts, last_exc))

    @staticmethod
    def _normalize(completion, attempts, elapsed_s, budget):
        choice = completion.choices[0]
        message = choice.message
        reasoning = getattr(message, "reasoning_content", None) or getattr(message, "reasoning", None)
        usage = None
        num_generated_tokens = None
        if getattr(completion, "usage", None) is not None:
            usage = completion.usage.model_dump()
            num_generated_tokens = usage.get("completion_tokens")
        return Response(
            generation=message.content or "",
            finish_reason=choice.finish_reason,
            reasoning_content=reasoning if isinstance(reasoning, str) and reasoning else None,
            usage=usage,
            num_generated_tokens=num_generated_tokens,
            attempts=attempts,
            elapsed_s=elapsed_s,
            context_budget=dict(budget),
        )
