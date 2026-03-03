#!/usr/bin/env python3
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

"""
Unified NeMo Inference Server with OpenAI-compatible API.

Backend-agnostic: all backend-specific logic lives in backend modules.
The server only knows about GenerationRequest/GenerationResult and the
InferenceBackend interface.

Exposes /v1/chat/completions endpoint for OpenAI compatibility.
Backends may register additional routes via get_extra_routes().
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .backends import GenerationRequest, GenerationResult, get_backend

# Debug flag
DEBUG = os.getenv("DEBUG", "").lower() in ("true", "1", "yes", "on")
logger = logging.getLogger(__name__)


@dataclass
class PendingRequest:
    """Container for a pending batched request."""

    request: GenerationRequest
    future: asyncio.Future
    timestamp: float


class RequestBatcher:
    """Manages request batching with configurable delay."""

    def __init__(self, backend, batch_size: int, batch_timeout: float):
        self.backend = backend
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pending_requests: List[PendingRequest] = []
        self.lock = asyncio.Lock()
        self.timeout_task: Optional[asyncio.Task] = None
        self.processing = False

        # Stats
        self.total_requests = 0
        self.total_batches = 0

    async def add_request(self, request: GenerationRequest) -> GenerationResult:
        """Add a request and wait for result."""
        future = asyncio.get_running_loop().create_future()
        pending = PendingRequest(request=request, future=future, timestamp=time.time())

        async with self.lock:
            self.pending_requests.append(pending)

            if len(self.pending_requests) >= self.batch_size:
                if DEBUG:
                    print(f"[Batcher] Batch full ({self.batch_size}), processing immediately")
                asyncio.create_task(self._process_batch())
            elif self.batch_timeout == 0:
                asyncio.create_task(self._process_batch())
            elif self.timeout_task is None or self.timeout_task.done():
                self.timeout_task = asyncio.create_task(self._timeout_handler())

        return await future

    async def _timeout_handler(self):
        """Handle batch timeout."""
        await asyncio.sleep(self.batch_timeout)
        async with self.lock:
            if self.pending_requests and not self.processing:
                if DEBUG:
                    print(f"[Batcher] Timeout, processing {len(self.pending_requests)} requests")
                asyncio.create_task(self._process_batch())

    async def _process_batch(self):
        """Process pending requests as a batch."""
        async with self.lock:
            if not self.pending_requests or self.processing:
                return

            self.processing = True
            batch = self.pending_requests[: self.batch_size]
            self.pending_requests = self.pending_requests[self.batch_size :]

        try:
            requests = [p.request for p in batch]

            if DEBUG:
                print(f"[Batcher] Processing batch of {len(requests)} requests")

            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, self.backend.generate, requests)

            if len(results) != len(batch):
                raise RuntimeError(f"Backend returned {len(results)} results for {len(batch)} requests")

            for pending, result in zip(batch, results, strict=True):
                if not pending.future.done():
                    pending.future.set_result(result)

            self.total_requests += len(batch)
            self.total_batches += 1

        except Exception as e:
            for pending in batch:
                if not pending.future.done():
                    pending.future.set_exception(e)
        finally:
            async with self.lock:
                self.processing = False
                if self.pending_requests:
                    if self.batch_timeout == 0 or len(self.pending_requests) >= self.batch_size:
                        asyncio.create_task(self._process_batch())
                    elif self.timeout_task is None or self.timeout_task.done():
                        self.timeout_task = asyncio.create_task(self._timeout_handler())


# Global state
backend_instance = None
request_batcher = None
server_config = {}


def extract_audio_from_messages(messages: List[Dict[str, Any]]) -> List[bytes]:
    """Extract all audio bytes from OpenAI-format messages.

    Supports these message content block formats:
    - {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,..."}}
    - {"type": "input_audio", "input_audio": {"data": "...", "format": "wav"}}
    """
    audio_list = []
    data_uri_pattern = re.compile(r"^data:audio/[^;]+;base64,(.+)$")

    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue

                try:
                    if item.get("type") == "audio_url":
                        audio_url = item.get("audio_url", {})
                        url = audio_url.get("url", "")
                        match = data_uri_pattern.match(url)
                        if match:
                            audio_list.append(base64.b64decode(match.group(1)))
                    elif item.get("type") == "input_audio":
                        input_audio = item.get("input_audio", {})
                        data = input_audio.get("data", "")
                        if data:
                            audio_list.append(base64.b64decode(data))
                except Exception as e:
                    if DEBUG:
                        print(f"[Server] Warning: Failed to decode audio block: {e}")
    return audio_list


def extract_text_from_messages(messages: List[Dict[str, Any]]) -> str:
    """Extract text content from OpenAI-format messages."""
    texts = []
    for message in messages:
        if message.get("role") == "system":
            continue
        content = message.get("content")
        if isinstance(content, str):
            if content:
                texts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        texts.append(text)
                elif isinstance(item, str):
                    texts.append(item)
    return " ".join(texts)


def extract_system_prompt(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Extract system prompt from messages."""
    for message in messages:
        if message.get("role") == "system":
            content = message.get("content")
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                texts = [
                    item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"
                ]
                return " ".join(texts) if texts else None
    return None


def create_app(
    backend_type: str,
    config_dict: Dict[str, Any],
    batch_size: int = 8,
    batch_timeout: float = 0.1,
) -> FastAPI:
    """Create and configure the FastAPI app.

    Args:
        backend_type: Name of the backend to use (e.g., 'magpie_tts').
        config_dict: Full configuration dict for the backend's config class.
        batch_size: Maximum batch size for request batching.
        batch_timeout: Seconds to wait before processing an incomplete batch.
    """
    global backend_instance, request_batcher, server_config

    app = FastAPI(
        title="Unified NeMo Inference Server",
        description=f"OpenAI-compatible API for NeMo model inference ({backend_type} backend)",
        version="1.0.0",
    )

    server_config = {
        "backend_type": backend_type,
        "model_path": config_dict.get("model_path", ""),
        "batch_size": batch_size,
        "batch_timeout": batch_timeout,
    }

    @app.on_event("startup")
    async def startup():
        global backend_instance, request_batcher

        # Look up backend class and its config class
        BackendClass = get_backend(backend_type)
        ConfigClass = BackendClass.get_config_class()

        # Validate and create config
        config = ConfigClass.from_dict(config_dict)

        # Instantiate and load backend
        print(f"[Server] Initializing {backend_type} backend...")
        backend_instance = BackendClass(config)
        backend_instance.load_model()

        # Create batcher
        request_batcher = RequestBatcher(backend_instance, batch_size, batch_timeout)

        # Register any extra routes from the backend
        extra_routes = BackendClass.get_extra_routes(backend_instance)
        for route in extra_routes:
            app.add_api_route(
                route["path"],
                route["endpoint"],
                methods=route.get("methods", ["GET"]),
            )
            print(f"[Server] Registered extra route: {route['path']}")

        print("[Server] Ready!")
        print(f"  Backend: {backend_type}")
        print(f"  Model: {config.model_path}")
        print(f"  Batch size: {batch_size}")
        print(f"  Batch timeout: {batch_timeout}s")

    @app.get("/")
    async def root():
        """Root endpoint with server info."""
        return {
            "service": "Unified NeMo Inference Server",
            "version": "1.0.0",
            "backend": server_config.get("backend_type"),
            "model": server_config.get("model_path"),
            "endpoints": ["/v1/chat/completions", "/health", "/v1/models"],
        }

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        if backend_instance is None:
            return JSONResponse(status_code=503, content={"status": "not_ready", "error": "Backend not initialized"})

        health_info = backend_instance.health_check()
        health_info["status"] = "healthy" if backend_instance.is_loaded else "not_ready"
        health_info["timestamp"] = datetime.now().isoformat()

        return health_info

    @app.get("/v1/models")
    async def list_models():
        """OpenAI-compatible models endpoint."""
        model_id = server_config.get("model_path", "unknown")
        return {
            "object": "list",
            "data": [
                {
                    "id": model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "nvidia",
                }
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Dict[str, Any]):
        """OpenAI-compatible chat completions endpoint with audio support."""
        if backend_instance is None or not backend_instance.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            messages = request.get("messages", [])
            if not messages:
                raise HTTPException(status_code=400, detail="No messages provided")

            # Extract components from messages
            audio_bytes_list = extract_audio_from_messages(messages)
            text = extract_text_from_messages(messages)
            system_prompt = extract_system_prompt(messages)

            # Get generation parameters
            max_tokens = request.get("max_tokens", 512)
            temperature = request.get("temperature", 1.0)
            top_p = request.get("top_p", 1.0)
            seed = request.get("seed")

            # Create generation request
            gen_request = GenerationRequest(
                text=text if text else None,
                system_prompt=system_prompt,
                audio_bytes=audio_bytes_list[0] if len(audio_bytes_list) == 1 else None,
                audio_bytes_list=audio_bytes_list if len(audio_bytes_list) > 1 else None,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                request_id=hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8],
                extra_params=request.get("extra_body", {}),
            )

            # Validate request
            error = backend_instance.validate_request(gen_request)
            if error:
                raise HTTPException(status_code=400, detail=error)

            # Process through batcher
            result = await request_batcher.add_request(gen_request)

            if not result.is_success():
                error_id = hashlib.md5(f"{time.time_ns()}".encode()).hexdigest()[:8]
                logger.error(
                    "Backend generation failed [error_id=%s, request_id=%s]: %s",
                    error_id,
                    gen_request.request_id,
                    result.error,
                )
                raise HTTPException(status_code=500, detail=f"Internal server error (error_id={error_id})")

            # Build OpenAI-compatible response
            response_id = f"chatcmpl-{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
            message_content = result.text or ""

            # Save outputs if AUDIO_SAVE_DIR is set
            save_dir = os.environ.get("AUDIO_SAVE_DIR", "")
            if save_dir:
                try:
                    os.makedirs(save_dir, exist_ok=True)
                except Exception as e:
                    error_id = hashlib.md5(f"{time.time_ns()}".encode()).hexdigest()[:8]
                    logger.error(
                        "Failed to prepare AUDIO_SAVE_DIR [error_id=%s, save_dir=%s]: %s",
                        error_id,
                        save_dir,
                        e,
                    )
                    raise HTTPException(status_code=500, detail=f"Internal server error (error_id={error_id})")
            saved_audio_path = None
            save_failures = []

            if save_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_filename = f"response_{timestamp}_{response_id}"

                try:
                    saved_json_path = os.path.join(save_dir, f"{base_filename}.json")
                    json_output = {
                        "response_id": response_id,
                        "timestamp": timestamp,
                        "text": message_content,
                        "debug_info": result.debug_info,
                        "generation_time_ms": result.generation_time_ms,
                        "num_tokens_generated": result.num_tokens_generated,
                    }
                    with open(saved_json_path, "w") as f:
                        json.dump(json_output, f, indent=2)
                except Exception as e:
                    save_failures.append(f"json:{type(e).__name__}:{e}")

                if result.audio_bytes:
                    try:
                        saved_audio_path = os.path.join(save_dir, f"{base_filename}.wav")
                        with open(saved_audio_path, "wb") as f:
                            f.write(result.audio_bytes)
                    except Exception as e:
                        save_failures.append(f"audio:{type(e).__name__}:{e}")

                if save_failures:
                    error_id = hashlib.md5(f"{time.time_ns()}".encode()).hexdigest()[:8]
                    logger.error(
                        "Failed to save response artifacts [error_id=%s, request_id=%s, response_id=%s, save_dir=%s, failures=%s]",
                        error_id,
                        gen_request.request_id,
                        response_id,
                        save_dir,
                        save_failures,
                    )
                    raise HTTPException(status_code=500, detail=f"Internal server error (error_id={error_id})")

            # Build audio output if available
            audio_output = None
            if result.audio_bytes:
                audio_output = {
                    "data": base64.b64encode(result.audio_bytes).decode("utf-8"),
                    "format": result.audio_format or "wav",
                    "sample_rate": result.audio_sample_rate,
                    "expires_at": int(time.time()) + 3600,
                    "transcript": result.text or "",
                }

            # Embed debug_info in content as JSON
            final_content = message_content
            if result.debug_info:
                final_content = f"{message_content}\n<debug_info>{json.dumps(result.debug_info)}</debug_info>"

            response = {
                "id": response_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": server_config.get("model_path"),
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": final_content,
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": -1,
                    "completion_tokens": result.num_tokens_generated or -1,
                    "total_tokens": -1,
                },
            }

            if audio_output:
                response["choices"][0]["message"]["audio"] = audio_output

            if result.debug_info:
                response["debug_info"] = result.debug_info

            if saved_audio_path:
                response["saved_audio_path"] = saved_audio_path

            return response

        except HTTPException:
            raise
        except Exception:
            error_id = hashlib.md5(f"{time.time_ns()}".encode()).hexdigest()[:8]
            logger.exception("Unhandled chat completion error [error_id=%s]", error_id)
            raise HTTPException(status_code=500, detail=f"Internal server error (error_id={error_id})")

    return app
