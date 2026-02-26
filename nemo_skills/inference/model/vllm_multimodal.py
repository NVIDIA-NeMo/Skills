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

"""VLLMMultimodalModel with support for audio input and output.

This module provides a multimodal model class that handles:
- Audio INPUT: encoding audio files to base64, chunking long audio
- Audio OUTPUT: saving audio responses from the server to disk
- External API support: NVIDIA Inference API, OpenAI, and other OpenAI-compatible APIs
"""

import base64
import copy
import json
import logging
import os
import re

from nemo_skills.utils import get_logger_name

from .audio_utils import (
    audio_file_to_base64,
    chunk_audio,
    load_audio_file,
    make_audio_content_block,
    save_audio_chunk_to_base64,
)
from .vllm import VLLMModel

LOG = logging.getLogger(get_logger_name(__file__))

# Pattern to extract debug_info from content
DEBUG_INFO_PATTERN = re.compile(r"\n?<debug_info>(.*?)</debug_info>", re.DOTALL)


class VLLMMultimodalModel(VLLMModel):
    """VLLMModel with support for audio input/output and external APIs.

    Audio INPUT capabilities:
    1. Converts audio file paths to base64-encoded input_audio format
    2. Chunks long audio files for models with duration limits
    3. Aggregates results from chunked audio processing

    Audio OUTPUT capabilities:
    1. Saves audio responses from the server to disk / output_dir/audio/
    2. Replaces the base64 data with the file path in the result

    Also supports external APIs (NVIDIA, OpenAI) via base_url parameter.

    Example usage:
        # Local vLLM server
        model = VLLMMultimodalModel(model="Qwen/Qwen2-Audio-7B")

        # NVIDIA Inference API
        model = VLLMMultimodalModel(
            model="gcp/google/gemini-2.5-pro",
            base_url="https://inference-api.nvidia.com/v1"
        )
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        enable_audio_chunking: bool = True,
        audio_chunk_task_types: list[str] | None = None,
        chunk_audio_threshold_sec: int = 30,
        audio_format: str | None = None,
        **kwargs,
    ):
        """Initialize VLLMMultimodalModel with audio I/O and external API support.

        Args:
            model: Model name (e.g., "Qwen/Qwen2-Audio-7B" for local, "gcp/google/gemini-2.5-pro" for NVIDIA API).
            base_url: API base URL. If None, defaults to local server. For external APIs, provide the full URL.
            enable_audio_chunking: Master switch for audio chunking.
            audio_chunk_task_types: If None, chunk all task types; if specified, only chunk these.
            chunk_audio_threshold_sec: Audio duration threshold for chunking (in seconds).
            audio_format: Format for audio content ("audio_url" or "input_audio"). If None, select by mode.
            **kwargs: Other parameters passed to VLLMModel/BaseModel.
        """
        super().__init__(model=model, base_url=base_url, **kwargs)

        # Determine if this is an external API (non-local URL)
        self._external_api_mode = not self._is_local_url(self.base_url)

        # Audio INPUT config
        self.enable_audio_chunking = enable_audio_chunking
        self.audio_chunk_task_types = audio_chunk_task_types
        self.chunk_audio_threshold_sec = chunk_audio_threshold_sec

        if audio_format is None:
            audio_format = "input_audio" if self._external_api_mode else "audio_url"
        if audio_format not in ("audio_url", "input_audio"):
            raise ValueError(f"Unsupported audio_format '{audio_format}'. Use 'audio_url' or 'input_audio'.")
        self.audio_format = audio_format

        # Audio OUTPUT config
        self.output_audio_dir = None
        if self.output_dir:
            self.output_audio_dir = os.path.join(self.output_dir, "audio")
            os.makedirs(self.output_audio_dir, exist_ok=True)
            LOG.info(f"Audio responses will be saved to: {self.output_audio_dir}")

    def _is_local_url(self, base_url: str | None) -> bool:
        """Check if the base_url points to a local server.

        Args:
            base_url: API base URL.

        Returns:
            True if local server, False otherwise.
        """
        if not base_url:
            return True  # No URL means local server (will use default host:port)
        local_patterns = ["127.0.0.1", "localhost", "0.0.0.0"]
        return any(pattern in base_url for pattern in local_patterns)

    def _get_api_key(self, api_key: str | None, api_key_env_var: str | None, base_url: str) -> str | None:
        """Get API key with smart detection for external APIs.

        Checks for API keys in the following order:
        1. Explicit api_key argument
        2. Environment variable specified by api_key_env_var
        3. Auto-detect based on base_url (NVIDIA_API_KEY, OPENAI_API_KEY, etc.)

        Args:
            api_key: Explicit API key.
            api_key_env_var: Environment variable name for API key.
            base_url: API base URL for auto-detection.

        Returns:
            API key string or None.
        """
        # First, try parent class logic (explicit key or env var)
        api_key = super()._get_api_key(api_key, api_key_env_var, base_url)

        if api_key is not None:
            return api_key

        # Auto-detect API key based on base_url
        if base_url:
            if "api.nvidia.com" in base_url or "inference-api.nvidia.com" in base_url:
                api_key = os.getenv("NV_INFERENCE_API_KEY") or os.getenv("NVIDIA_API_KEY")
                if not api_key:
                    raise ValueError(
                        "NV_INFERENCE_API_KEY or NVIDIA_API_KEY is required for NVIDIA APIs and could not be found. "
                        "Set NV_INFERENCE_API_KEY/NVIDIA_API_KEY environment variable or pass api_key explicitly."
                    )
                return api_key

            if "api.openai.com" in base_url:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError(
                        "OPENAI_API_KEY is required for OpenAI APIs and could not be found. "
                        "Set OPENAI_API_KEY environment variable or pass api_key explicitly."
                    )
                return api_key

            if "generativelanguage.googleapis.com" in base_url:
                api_key = os.getenv("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError(
                        "GOOGLE_API_KEY is required for Google APIs and could not be found. "
                        "Set GOOGLE_API_KEY environment variable or pass api_key explicitly."
                    )
                return api_key

        return api_key

    def _build_request_body(self, top_k, min_p, repetition_penalty, extra_body: dict | None = None):
        """Build request body, skipping vLLM-specific params for external APIs.

        Args:
            top_k: Top-k sampling parameter (vLLM, default -1).
            min_p: Min-p sampling parameter (vLLM, default 0.0).
            repetition_penalty: Repetition penalty parameter (vLLM, default 1.0).
            extra_body: Additional parameters to include.

        Returns:
            Dictionary of extra body parameters for the request.

        Raises:
            ValueError: If vLLM-specific params are set to non-default values in external API mode.
        """
        # For external APIs, fail if user explicitly set vLLM-specific parameters
        if self._external_api_mode:
            non_default_params = []
            if top_k != -1:
                non_default_params.append(f"top_k={top_k}")
            if min_p != 0.0:
                non_default_params.append(f"min_p={min_p}")
            if repetition_penalty != 1.0:
                non_default_params.append(f"repetition_penalty={repetition_penalty}")

            if non_default_params:
                raise ValueError(
                    f"vLLM-specific parameters are not supported for external APIs: {', '.join(non_default_params)}. "
                    "These parameters only work with local vLLM servers."
                )
            return extra_body or {}

        # For local vLLM server, use full parameter set
        return super()._build_request_body(top_k, min_p, repetition_penalty, extra_body=extra_body)

    def _parse_chat_completion_response(self, response, include_response: bool = False, **kwargs) -> dict:
        """Parse chat completion response and save any audio to disk."""
        result = super()._parse_chat_completion_response(response, include_response=include_response, **kwargs)

        # Extract debug_info from content (embedded as JSON in <debug_info> tags)
        if "generation" in result and result["generation"]:
            match = DEBUG_INFO_PATTERN.search(result["generation"])
            if match:
                try:
                    result["debug_info"] = json.loads(match.group(1))
                    # Strip debug_info from generation
                    result["generation"] = DEBUG_INFO_PATTERN.sub("", result["generation"])
                except json.JSONDecodeError:
                    LOG.warning("Failed to parse debug_info JSON from content")

        choice = response.choices[0]
        if hasattr(choice.message, "audio") and choice.message.audio:
            audio_result = self._process_audio_response(choice.message.audio, response.id)
            result["audio"] = audio_result

        # Strip audio data from serialized_output to avoid duplication
        if "serialized_output" in result:
            for item in result["serialized_output"]:
                if isinstance(item, dict) and "audio" in item:
                    # Keep only metadata, remove base64 data
                    if isinstance(item["audio"], dict) and "data" in item["audio"]:
                        del item["audio"]["data"]
                # Also strip debug_info from serialized content
                if isinstance(item, dict) and "content" in item and item["content"]:
                    item["content"] = DEBUG_INFO_PATTERN.sub("", item["content"])

        return result

    def _process_audio_response(self, audio_data, response_id: str) -> dict:
        """Process audio data: save to file and return metadata with path."""
        audio_info = {
            "format": getattr(audio_data, "format", "wav"),
            "sample_rate": getattr(audio_data, "sample_rate", 22050),
            "transcript": getattr(audio_data, "transcript", None),
        }

        audio_base64 = getattr(audio_data, "data", None)
        if not audio_base64:
            return audio_info

        if self.output_audio_dir:
            try:
                audio_bytes = base64.b64decode(audio_base64)
                filename = f"{response_id}.wav"
                filepath = os.path.join(self.output_audio_dir, filename)

                with open(filepath, "wb") as f:
                    f.write(audio_bytes)

                audio_info["path"] = filepath
                audio_info["size_bytes"] = len(audio_bytes)
                LOG.info(f"Saved audio: {filepath} ({len(audio_bytes)} bytes)")
            except Exception as e:
                LOG.warning(f"Failed to save audio: {e}")
                audio_info["data"] = audio_base64
        else:
            audio_info["data"] = audio_base64

        return audio_info

    # =====================
    # Audio INPUT methods
    # =====================

    def _preprocess_messages_for_model(self, messages: list[dict]) -> list[dict]:
        """Preprocess messages - creates copies to avoid mutation.

        Note: /no_think suffix is passed through unchanged (handled by the model).

        Args:
            messages: List of message dicts.

        Returns:
            Copy of message dicts.
        """
        return [copy.deepcopy(msg) for msg in messages]

    def content_text_to_list(self, message: dict) -> dict:
        """Convert message content with audio to proper list format.

        Handles 'audio' or 'audios' keys in messages and converts them to
        base64-encoded input_audio content items.

        CRITICAL: Audio must come BEFORE text for models to process correctly.

        Args:
            message: Message dict that may contain 'audio' or 'audios' fields.

        Returns:
            New message dict with content converted to list format including audio.
        """
        if "audio" not in message and "audios" not in message:
            return copy.deepcopy(message)

        result = copy.deepcopy(message)

        if "content" not in result:
            raise KeyError("Missing required 'content' in message")
        content = result["content"]
        if isinstance(content, str):
            result["content"] = [{"type": "text", "text": content}]
        elif not isinstance(content, list):
            raise TypeError(f"Unexpected content type: {type(content)}")

        audio_items = []

        if "audio" in result:
            audio = result.pop("audio")
            audio_path = os.path.join(self.data_dir, audio["path"])
            base64_audio = audio_file_to_base64(audio_path)
            audio_items.append(make_audio_content_block(base64_audio, self.audio_format))
        elif "audios" in result:
            for audio in result.pop("audios"):
                audio_path = os.path.join(self.data_dir, audio["path"])
                base64_audio = audio_file_to_base64(audio_path)
                audio_items.append(make_audio_content_block(base64_audio, self.audio_format))

        if audio_items:
            result["content"] = audio_items + result["content"]

        return result

    def _needs_audio_chunking(self, messages: list[dict], task_type: str = None) -> tuple[bool, str, float]:
        """Check if audio in messages needs chunking.

        Args:
            messages: List of message dicts.
            task_type: Optional task type for chunking filtering.

        Returns:
            Tuple of (needs_chunking, audio_path, duration).
        """
        if not self.enable_audio_chunking:
            return False, None, 0.0

        # Check if task type should be chunked (if filter is specified)
        if self.audio_chunk_task_types is not None:
            if task_type not in self.audio_chunk_task_types:
                return False, None, 0.0

        # Find audio in messages
        for msg in messages:
            if msg["role"] == "user":
                if "audio" in msg:
                    audio_info = msg["audio"]
                elif "audios" in msg:
                    audios = msg["audios"]
                    audio_info = audios[0] if audios else {}
                else:
                    continue
                if audio_info and "path" in audio_info:
                    audio_path = os.path.join(self.data_dir, audio_info["path"])

                    if not os.path.exists(audio_path):
                        return False, None, 0.0

                    # Load audio to check duration
                    audio_array, sampling_rate = load_audio_file(audio_path)
                    duration = len(audio_array) / sampling_rate

                    if duration > self.chunk_audio_threshold_sec:
                        return True, audio_path, duration

        return False, None, 0.0

    async def _generate_with_chunking(
        self,
        messages: list[dict],
        audio_path: str,
        duration: float,
        tokens_to_generate: int | None = None,
        **kwargs,
    ) -> dict:
        """Generate by chunking long audio and aggregating results.

        Args:
            messages: Original messages containing audio reference.
            audio_path: Path to the audio file to chunk.
            duration: Duration of audio in seconds.
            tokens_to_generate: Max tokens per chunk.
            **kwargs: Additional generation parameters.

        Returns:
            Aggregated result with combined generation from all chunks.
        """
        audio_array, sampling_rate = load_audio_file(audio_path)
        chunks = chunk_audio(audio_array, sampling_rate, self.chunk_audio_threshold_sec)

        LOG.info(f"Chunking audio ({duration:.1f}s) into {len(chunks)} chunks of {self.chunk_audio_threshold_sec}s")

        if not chunks:
            raise RuntimeError("No audio chunks generated - audio may be too short or invalid")

        chunk_results = []
        result = None

        # Track cumulative statistics across chunks
        total_num_generated_tokens = 0

        for chunk_idx, audio_chunk in enumerate(chunks):
            chunk_messages = []

            for msg in messages:
                msg_copy = copy.deepcopy(msg)

                if msg_copy["role"] == "user" and ("audio" in msg_copy or "audios" in msg_copy):
                    chunk_base64 = save_audio_chunk_to_base64(audio_chunk, sampling_rate)

                    if "content" not in msg_copy:
                        raise KeyError("Missing required 'content' in message")
                    content = msg_copy["content"]
                    if isinstance(content, str):
                        text_content = [{"type": "text", "text": content}]
                    else:
                        text_content = content

                    # Add audio chunk at the beginning (before text)
                    msg_copy["content"] = [make_audio_content_block(chunk_base64, self.audio_format)] + text_content

                    # Remove original audio fields
                    msg_copy.pop("audio", None)
                    msg_copy.pop("audios", None)

                chunk_messages.append(msg_copy)

            # Generate for this chunk using parent's generate_async
            result = await super().generate_async(
                prompt=chunk_messages, tokens_to_generate=tokens_to_generate, **kwargs
            )

            # Sum statistics from each chunk
            total_num_generated_tokens += result["num_generated_tokens"]

            generation = result["generation"]
            chunk_results.append(generation.strip())

        # Aggregate results
        aggregated_text = " ".join(chunk_results)

        final_result = result.copy()
        final_result["generation"] = aggregated_text
        final_result["num_audio_chunks"] = len(chunks)
        final_result["audio_duration"] = duration
        # Update with summed statistics
        final_result["num_generated_tokens"] = total_num_generated_tokens

        return final_result

    async def generate_async(
        self,
        prompt: str | list[dict] | None = None,
        tokens_to_generate: int | None = None,
        task_type: str | None = None,
        **kwargs,
    ) -> dict:
        """Generate with automatic audio chunking for long audio files.

        This override checks if the prompt (messages) contains long audio.
        If so, it chunks the audio, processes each chunk separately, and aggregates results.

        Args:
            prompt: Either a string (text completion) or list of messages (chat).
            tokens_to_generate: Max tokens to generate.
            task_type: Optional task type for chunking filtering.
            **kwargs: Additional arguments passed to the underlying model.

        Returns:
            Generation result dict with 'generation' key and optional metadata.
        """
        if isinstance(prompt, list):
            messages = prompt
            needs_chunking, audio_path, duration = self._needs_audio_chunking(messages, task_type)

            if needs_chunking:
                return await self._generate_with_chunking(messages, audio_path, duration, tokens_to_generate, **kwargs)

            # No chunking needed - convert audio fields to base64 format
            messages = [self.content_text_to_list(msg) for msg in messages]
            prompt = messages

        # Call parent's generate_async (which handles audio OUTPUT via _parse_chat_completion_response)
        return await super().generate_async(prompt=prompt, tokens_to_generate=tokens_to_generate, **kwargs)
