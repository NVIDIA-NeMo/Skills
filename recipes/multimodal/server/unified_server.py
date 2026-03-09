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

Supports multiple NeMo model backends:
- SALM: Speech-Augmented Language Model
- TTS: Text-to-Speech (MagpieTTS)
- S2S: Speech-to-Speech (Duplex)

Exposes only /v1/chat/completions endpoint for OpenAI compatibility.

Usage:
    python unified_server.py --backend s2s --model /path/to/model
"""

import asyncio
import base64
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .backends import BackendConfig, GenerationRequest, GenerationResult, get_backend
from .session_manager import SessionManager

# Configuration from environment
HOST = os.getenv("UNIFIED_SERVER_HOST", "0.0.0.0")
PORT = int(os.getenv("UNIFIED_SERVER_PORT", "8000"))
BACKEND_TYPE = os.getenv("UNIFIED_SERVER_BACKEND", "salm")
MODEL_PATH = os.getenv("UNIFIED_SERVER_MODEL_PATH", "")
CODEC_MODEL_PATH = os.getenv("UNIFIED_SERVER_CODEC_MODEL_PATH", "")

# Batching configuration
# Note: S2S backends process requests sequentially anyway, so batch_size>1 just adds delay
# Use batch_timeout=0 for immediate processing without waiting
BATCH_SIZE = int(os.getenv("UNIFIED_SERVER_BATCH_SIZE", "1"))
BATCH_TIMEOUT = float(os.getenv("UNIFIED_SERVER_BATCH_TIMEOUT", "0"))

# Generation defaults
MAX_NEW_TOKENS = int(os.getenv("UNIFIED_SERVER_MAX_NEW_TOKENS", "512"))
TEMPERATURE = float(os.getenv("UNIFIED_SERVER_TEMPERATURE", "1.0"))
TOP_P = float(os.getenv("UNIFIED_SERVER_TOP_P", "1.0"))

# Debug
DEBUG = os.getenv("DEBUG", "").lower() in ("true", "1", "yes", "on")
INCLUDE_DEBUG_INFO = os.getenv("INCLUDE_DEBUG_INFO", "true").lower() in ("true", "1", "yes", "on")


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
        future = asyncio.Future()
        pending = PendingRequest(request=request, future=future, timestamp=time.time())

        async with self.lock:
            self.pending_requests.append(pending)

            # Check if we should process immediately
            if len(self.pending_requests) >= self.batch_size:
                if DEBUG:
                    print(f"[Batcher] Batch full ({self.batch_size}), processing immediately")
                asyncio.create_task(self._process_batch())
            elif self.batch_timeout == 0:
                # No delay mode
                asyncio.create_task(self._process_batch())
            elif self.timeout_task is None or self.timeout_task.done():
                # Schedule timeout
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
            # Extract requests
            requests = [p.request for p in batch]

            if DEBUG:
                print(f"[Batcher] Processing batch of {len(requests)} requests")

            # Run inference in thread pool to not block event loop
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(None, self.backend.generate, requests)

            # Complete futures
            for pending, result in zip(batch, results):
                if not pending.future.done():
                    pending.future.set_result(result)

            # Update stats
            self.total_requests += len(batch)
            self.total_batches += 1

        except Exception as e:
            # Set exception for all pending requests
            for pending in batch:
                if not pending.future.done():
                    pending.future.set_exception(e)
        finally:
            async with self.lock:
                self.processing = False
                # Process more if pending
                if self.pending_requests:
                    if self.batch_timeout == 0 or len(self.pending_requests) >= self.batch_size:
                        asyncio.create_task(self._process_batch())
                    elif self.timeout_task is None or self.timeout_task.done():
                        self.timeout_task = asyncio.create_task(self._timeout_handler())


# Global state
backend_instance = None
request_batcher = None
session_manager = None
server_config = {}


def compute_session_hash(messages: List[Dict[str, Any]]) -> str:
    """Compute deterministic hash from messages for session identification.

    Used for automatic session management without client-provided session_id.
    Hash is based on message roles and content (text only, not audio data).
    """
    hash_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        # For list content, extract text parts only
        if isinstance(content, list):
            text_parts = [
                item.get("text", "") for item in content if isinstance(item, dict) and item.get("type") == "text"
            ]
            content = " ".join(text_parts)
        hash_parts.append(f"{role}:{content}")

    combined = "|".join(hash_parts)
    return hashlib.md5(combined.encode()).hexdigest()[:16]


def extract_audio_from_messages(messages: List[Dict[str, Any]]) -> List[bytes]:
    """Extract all audio bytes from OpenAI-format messages.

    Supports two formats:
    1. audio_url: {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,..."}}
    2. input_audio: {"type": "input_audio", "input_audio": {"data": "...", "format": "wav"}}

    Returns a list of audio bytes (one per audio found), preserving message order.
    """
    audio_list = []
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    # Format 1: audio_url (data URL style)
                    if item.get("type") == "audio_url":
                        audio_url = item.get("audio_url", {})
                        url = audio_url.get("url", "")
                        match = re.match(r"data:audio/\w+;base64,(.+)", url)
                        if match:
                            audio_list.append(base64.b64decode(match.group(1)))

                    # Format 2: input_audio (OpenAI native style)
                    elif item.get("type") == "input_audio":
                        input_audio = item.get("input_audio", {})
                        data = input_audio.get("data", "")
                        if data:
                            audio_list.append(base64.b64decode(data))
    return audio_list


def extract_text_from_messages(messages: List[Dict[str, Any]]) -> str:
    """Extract text content from OpenAI-format messages."""
    texts = []
    for message in messages:
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
    backend_type: str = BACKEND_TYPE,
    model_path: str = MODEL_PATH,
    codec_model_path: str = CODEC_MODEL_PATH,
    batch_size: int = BATCH_SIZE,
    batch_timeout: float = BATCH_TIMEOUT,
    device: str = "cuda",
    dtype: str = "bfloat16",
    extra_config: Dict[str, Any] = None,
) -> FastAPI:
    """Create and configure the FastAPI app."""
    global backend_instance, request_batcher, session_manager, server_config

    # Extract server-level config from extra_config
    ignore_system_prompt = extra_config.pop("ignore_system_prompt", False) if extra_config else False
    override_system_prompt = extra_config.get("system_prompt", None) if extra_config else None
    session_ttl = extra_config.pop("session_ttl", 300.0) if extra_config else 300.0
    max_sessions = extra_config.pop("max_sessions", 100) if extra_config else 100

    app = FastAPI(
        title="Unified NeMo Inference Server",
        description=f"OpenAI-compatible API for NeMo model inference ({backend_type} backend)",
        version="1.0.0",
    )

    # Store config
    server_config = {
        "backend_type": backend_type,
        "model_path": model_path,
        "codec_model_path": codec_model_path,
        "batch_size": batch_size,
        "batch_timeout": batch_timeout,
        "device": device,
        "dtype": dtype,
        "ignore_system_prompt": ignore_system_prompt,
        "override_system_prompt": override_system_prompt,
        "session_ttl": session_ttl,
        "max_sessions": max_sessions,
    }

    @app.on_event("startup")
    async def startup():
        global backend_instance, request_batcher, session_manager

        # Build backend config
        config_dict = {
            "model_path": model_path,
            "device": device,
            "dtype": dtype,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        }

        # Add backend-specific config
        if codec_model_path:
            config_dict["codec_model_path"] = codec_model_path

        if extra_config:
            config_dict.update(extra_config)

        config = BackendConfig.from_dict(config_dict)

        # Get and instantiate backend
        print(f"[Server] Initializing {backend_type} backend...")
        BackendClass = get_backend(backend_type)
        backend_instance = BackendClass(config)

        # Load model
        backend_instance.load_model()

        # Create batcher
        request_batcher = RequestBatcher(backend_instance, batch_size, batch_timeout)

        # Initialize session manager for session-aware backends
        if backend_type == "s2s_session":
            session_manager = SessionManager(ttl_seconds=session_ttl, max_sessions=max_sessions)
            print(f"[Server] Session manager initialized (TTL: {session_ttl}s, max: {max_sessions})")

            # Warmup inference to pre-compile Triton kernels (avoids race conditions on first requests)
            print("[Server] Running warmup inference to compile Triton kernels...")
            try:
                backend_instance.warmup()
                print("[Server] Warmup complete - Triton kernels compiled")
            except Exception as e:
                print(f"[Server] Warmup failed (will compile on first request): {e}")

        print("[Server] Ready!")
        print(f"  Backend: {backend_type}")
        print(f"  Model: {model_path}")
        print(f"  Batch size: {batch_size}")
        print(f"  Batch timeout: {batch_timeout}s")
        if override_system_prompt:
            print(f"  System prompt: OVERRIDE with '{override_system_prompt[:80]}...'")
        elif ignore_system_prompt:
            print("  System prompts: IGNORED")

    @app.get("/")
    async def root():
        """Root endpoint with server info."""
        endpoints = ["/v1/chat/completions", "/health"]
        if backend_type == "s2s_session":
            endpoints.extend(["/v1/sessions", "/v1/sessions/{session_id}"])
        return {
            "service": "Unified NeMo Inference Server",
            "version": "1.0.0",
            "backend": server_config.get("backend_type"),
            "model": server_config.get("model_path"),
            "endpoints": endpoints,
        }

    # Session management endpoints (only for s2s_session backend)
    @app.get("/v1/sessions")
    async def list_sessions():
        """List all active sessions."""
        if session_manager is None:
            raise HTTPException(status_code=404, detail="Session management not enabled for this backend")
        return {
            "sessions": session_manager.list_sessions(),
            "count": len(session_manager),
            "ttl_seconds": session_manager.ttl_seconds,
        }

    @app.get("/v1/sessions/{session_id}")
    async def get_session(session_id: str):
        """Get session info."""
        if session_manager is None:
            raise HTTPException(status_code=404, detail="Session management not enabled for this backend")
        info = session_manager.get_session_info(session_id)
        if info is None:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        return info

    @app.delete("/v1/sessions/{session_id}")
    async def delete_session(session_id: str):
        """Delete a session and generate final session audio."""
        if session_manager is None:
            raise HTTPException(status_code=404, detail="Session management not enabled for this backend")

        # Get session state before deleting
        session_state = session_manager.get_session(session_id)
        if session_state is None:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

        # Call on_session_close to generate session audio
        close_result = {}
        if backend_instance is not None and hasattr(backend_instance, "on_session_close"):
            try:
                close_result = backend_instance.on_session_close(session_state)
            except Exception as e:
                print(f"[Server] Error in on_session_close: {e}")
                import traceback

                traceback.print_exc()

        # Now delete the session
        session_manager.delete_session(session_id)

        return {"success": True, "session_id": session_id, **close_result}

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
        model_id = server_config.get("model_path", "unknown") if server_config else "unknown"
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
        """OpenAI-compatible chat completions endpoint with audio support.

        Accepts messages in OpenAI format with audio_url for audio content:
        {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": [
                    {"type": "text", "text": "..."},
                    {"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,..."}}
                ]}
            ],
            "max_tokens": 512,
            "temperature": 1.0,
            "extra_body": {"session_id": "optional-session-id"}
        }
        """
        if backend_instance is None or not backend_instance.is_loaded:
            raise HTTPException(status_code=503, detail="Model not loaded")

        try:
            messages = request.get("messages", [])
            if not messages:
                raise HTTPException(status_code=400, detail="No messages provided")

            # Extract session_id from extra_body (for s2s_session backend)
            extra_body = request.get("extra_body", {})
            session_id = extra_body.get("session_id") if isinstance(extra_body, dict) else None

            # Extract components from messages
            audio_bytes_list = extract_audio_from_messages(messages)
            text = extract_text_from_messages(messages)
            system_prompt = extract_system_prompt(messages)

            # CLI --system_prompt overrides per-request prompt from messages
            if server_config.get("override_system_prompt"):
                system_prompt = server_config["override_system_prompt"]
            # CLI --ignore_system_prompt nullifies it entirely
            elif server_config.get("ignore_system_prompt", False):
                system_prompt = None

            # Get generation parameters
            max_tokens = request.get("max_tokens", MAX_NEW_TOKENS)
            temperature = request.get("temperature", TEMPERATURE)
            top_p = request.get("top_p", TOP_P)
            seed = request.get("seed")

            # Create generation request
            # For s2s_session: use last audio only (current turn) - history is in KV cache
            # For other backends: use audio_bytes_list for multi-turn support
            if backend_type == "s2s_session":
                # Session backend only needs current turn's audio (last in list)
                current_audio = audio_bytes_list[-1] if audio_bytes_list else None
                gen_request = GenerationRequest(
                    text=text if text else None,
                    system_prompt=system_prompt,
                    audio_bytes=current_audio,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                    request_id=hashlib.md5(f"{time.time()}".encode()).hexdigest()[:8],
                )
            else:
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
                )

            # Validate request
            error = backend_instance.validate_request(gen_request)
            if error:
                raise HTTPException(status_code=400, detail=error)

            # Handle s2s_session backend with session support
            if backend_type == "s2s_session" and session_manager is not None:
                # Compute session IDs from message history (OpenAI API compatible)
                # restore_session_id: hash of messages before last (assistant, user) pair
                # save_session_id: hash of all messages (full conversation)
                non_system_messages = [m for m in messages if m.get("role") != "system"]

                if len(non_system_messages) <= 1:
                    # First turn: no previous session to restore
                    restore_session_id = None
                else:
                    # Multi-turn: restore from previous conversation state
                    # messages[:-2] excludes last assistant + last user
                    prefix_messages = messages[:-2] if len(messages) >= 2 else []
                    restore_session_id = compute_session_hash(prefix_messages) if prefix_messages else None

                # Session to save after this turn (full conversation)
                save_session_id = compute_session_hash(messages)

                # Use client-provided session_id if available, otherwise use computed one
                effective_restore_id = session_id if session_id else restore_session_id

                # Get or create session
                session_state = session_manager.get_or_create_session(effective_restore_id)

                # Run inference with session in thread pool
                loop = asyncio.get_event_loop()
                result, updated_session = await loop.run_in_executor(
                    None,
                    backend_instance.generate_with_session,
                    gen_request,
                    session_state,
                )

                # Save updated session state under the new hash (full conversation)
                if updated_session is not None:
                    session_manager.save_session(save_session_id, updated_session)
                session_id = save_session_id
            else:
                # Process through batcher (non-session path)
                result = await request_batcher.add_request(gen_request)
                session_id = None

            if not result.is_success():
                raise HTTPException(status_code=500, detail=result.error)

            # Build OpenAI-compatible response
            response_id = f"chatcmpl-{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"

            # Build message content
            message_content = result.text or ""

            # Save outputs to files before sending response (in case client times out)
            import json as json_lib
            import os
            from datetime import datetime

            save_dir = os.environ.get(
                "AUDIO_SAVE_DIR", "/lustre/fsw/portfolios/llmservice/users/vmendelev/tmp/voicebench_test"
            )
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_filename = f"response_{timestamp}_{response_id}"

            saved_audio_path = None
            saved_json_path = None

            # Save JSON with text and debug info
            try:
                saved_json_path = os.path.join(save_dir, f"{base_filename}.json")
                json_output = {
                    "response_id": response_id,
                    "timestamp": timestamp,
                    "text": message_content,
                    "asr_text": result.asr_text,
                    "debug_info": result.debug_info,
                    "generation_time_ms": result.generation_time_ms,
                    "num_tokens_generated": result.num_tokens_generated,
                }
                with open(saved_json_path, "w") as f:
                    json_lib.dump(json_output, f, indent=2)
                print(f"[Server] JSON saved to: {saved_json_path}")
            except Exception as e:
                print(f"[Server] Warning: Failed to save JSON: {e}")

            # Include audio output if available (base64 encoded)
            audio_output = None
            if result.audio_bytes:
                # Save audio file
                try:
                    saved_audio_path = os.path.join(save_dir, f"{base_filename}.wav")
                    with open(saved_audio_path, "wb") as f:
                        f.write(result.audio_bytes)
                    print(f"[Server] Audio saved to: {saved_audio_path} ({len(result.audio_bytes)} bytes)")
                except Exception as e:
                    print(f"[Server] Warning: Failed to save audio: {e}")

                audio_output = {
                    "data": base64.b64encode(result.audio_bytes).decode("utf-8"),
                    "format": result.audio_format or "wav",
                    "sample_rate": result.audio_sample_rate,
                    "expires_at": int(time.time()) + 3600,  # 1 hour expiry
                    "transcript": result.text or "",  # Text transcript of the audio
                }

            # Inject effective_system_prompt into debug_info before embedding
            if result.debug_info:
                result.debug_info["effective_system_prompt"] = system_prompt

            # Embed debug_info in content as JSON (OpenAI-compatible)
            final_content = message_content
            if result.debug_info and INCLUDE_DEBUG_INFO:
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

            # Add audio to response if available
            if audio_output:
                response["choices"][0]["message"]["audio"] = audio_output

            # Add ASR text (user speech transcription) if available
            if result.asr_text is not None:
                response["choices"][0]["message"]["asr_text"] = result.asr_text

            # Add debug info at top level too (for non-litellm clients)
            if result.debug_info and INCLUDE_DEBUG_INFO:
                response["debug_info"] = result.debug_info

            # Add saved file paths if available
            if saved_audio_path:
                response["saved_audio_path"] = saved_audio_path
            if saved_json_path:
                response["saved_json_path"] = saved_json_path

            # Add session_id for session-aware backends
            if session_id:
                response["session_id"] = session_id

            return response

        except HTTPException:
            raise
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))

    return app


def main():
    """Run the server from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Unified NeMo Inference Server")
    parser.add_argument(
        "--backend",
        default=BACKEND_TYPE,
        choices=["salm", "tts", "s2s", "s2s_incremental", "s2s_session"],
        help="Backend type to use",
    )
    parser.add_argument("--model", default=MODEL_PATH, help="Path to model")
    parser.add_argument("--codec_model", default=CODEC_MODEL_PATH, help="Path to codec model (for TTS/S2S)")
    parser.add_argument("--host", default=HOST, help="Server host")
    parser.add_argument("--port", type=int, default=PORT, help="Server port")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument(
        "--batch_timeout", type=float, default=BATCH_TIMEOUT, help="Batch timeout in seconds (0 for no delay)"
    )
    parser.add_argument("--device", default="cuda", help="Device to use")
    parser.add_argument("--dtype", default="bfloat16", help="Model dtype")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Backend-specific arguments
    parser.add_argument("--prompt_format", default=None, help="Prompt format (SALM)")
    parser.add_argument("--phoneme_input_type", default="predicted", help="Phoneme input type (TTS)")
    parser.add_argument("--decoder_only_model", action="store_true", help="Use decoder-only model (TTS)")
    parser.add_argument(
        "--ignore_system_prompt",
        action="store_true",
        help="Ignore system prompts from requests (for models that don't support them)",
    )
    parser.add_argument(
        "--silence_padding_sec",
        type=float,
        default=5.0,
        help="Seconds of silence to append after audio (S2S backend)",
    )

    # S2S Incremental backend arguments
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to YAML config file (s2s_incremental backend)",
    )
    parser.add_argument(
        "--llm_checkpoint_path",
        type=str,
        default=None,
        help="Path to LLM checkpoint (s2s_incremental backend)",
    )
    parser.add_argument(
        "--tts_checkpoint_path",
        type=str,
        default=None,
        help="Path to TTS checkpoint (s2s_incremental backend)",
    )
    parser.add_argument(
        "--speaker_reference",
        type=str,
        default=None,
        help="Path to speaker reference audio for TTS (s2s_incremental backend)",
    )
    parser.add_argument(
        "--num_frames_per_inference",
        type=int,
        default=1,
        help="Frames per inference step (s2s_incremental backend)",
    )
    parser.add_argument(
        "--decode_audio",
        action="store_true",
        default=True,
        help="Enable audio output via TTS (s2s_incremental backend)",
    )
    parser.add_argument(
        "--no_decode_audio",
        action="store_true",
        help="Disable audio output (s2s_incremental backend)",
    )

    args = parser.parse_args()

    if args.debug:
        global DEBUG
        DEBUG = True

    # Build extra config from backend-specific args
    extra_config = {}
    if args.prompt_format:
        extra_config["prompt_format"] = args.prompt_format
    if args.phoneme_input_type:
        extra_config["phoneme_input_type"] = args.phoneme_input_type
    if args.decoder_only_model:
        extra_config["decoder_only_model"] = True
    if args.silence_padding_sec != 5.0:  # Only add if different from default
        extra_config["silence_padding_sec"] = args.silence_padding_sec
    extra_config["ignore_system_prompt"] = args.ignore_system_prompt

    # S2S Incremental backend config
    if args.config_path:
        extra_config["config_path"] = args.config_path
    if args.llm_checkpoint_path:
        extra_config["llm_checkpoint_path"] = args.llm_checkpoint_path
    if args.tts_checkpoint_path:
        extra_config["tts_checkpoint_path"] = args.tts_checkpoint_path
    if args.speaker_reference:
        extra_config["speaker_reference"] = args.speaker_reference
    if args.num_frames_per_inference != 1:
        extra_config["num_frames_per_inference"] = args.num_frames_per_inference
    if args.no_decode_audio:
        extra_config["decode_audio"] = False
    elif args.decode_audio:
        extra_config["decode_audio"] = True

    app = create_app(
        backend_type=args.backend,
        model_path=args.model,
        codec_model_path=args.codec_model,
        batch_size=args.batch_size,
        batch_timeout=args.batch_timeout,
        device=args.device,
        dtype=args.dtype,
        extra_config=extra_config if extra_config else None,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
