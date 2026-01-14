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
    save_audio_chunk_to_base64,
)
from .vllm import VLLMModel

LOG = logging.getLogger(get_logger_name(__file__))

# Pattern to extract debug_info from content
DEBUG_INFO_PATTERN = re.compile(r"\n?<debug_info>(.*?)</debug_info>", re.DOTALL)


class VLLMMultimodalModel(VLLMModel):
    """VLLMModel with support for audio input and output.

    Audio INPUT capabilities:
    1. Converts audio file paths to base64-encoded audio_url format
    2. Chunks long audio files for models with duration limits
    3. Aggregates results from chunked audio processing

    Audio OUTPUT capabilities:
    1. Saves audio responses from the server to disk / output_dir/audio/
    2. Replaces the base64 data with the file path in the result

    """

    def __init__(
        self,
        enable_audio_chunking: bool = True,
        audio_chunk_task_types: list[str] | None = None,
        chunk_audio_threshold_sec: int = 30,
        **kwargs,
    ):
        """Initialize VLLMMultimodalModel with audio I/O support.

        Args:
            enable_audio_chunking: Master switch for audio chunking.
            audio_chunk_task_types: If None, chunk all task types; if specified, only chunk these.
            chunk_audio_threshold_sec: Audio duration threshold for chunking (in seconds).
            **kwargs: Other parameters passed to VLLMModel/BaseModel.
        """
        super().__init__(**kwargs)

        # Audio INPUT config
        self.enable_audio_chunking = enable_audio_chunking
        self.audio_chunk_task_types = audio_chunk_task_types
        self.chunk_audio_threshold_sec = chunk_audio_threshold_sec

        # Audio OUTPUT config
        self.output_audio_dir = None
        if self.output_dir:
            self.output_audio_dir = os.path.join(self.output_dir, "audio")
            os.makedirs(self.output_audio_dir, exist_ok=True)
            LOG.info(f"Audio responses will be saved to: {self.output_audio_dir}")

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

    def content_text_to_list(self, message: dict) -> dict:
        """Convert message content with audio to proper list format.

        Handles 'audio' or 'audios' keys in messages and converts them to
        base64-encoded audio_url content items.

        CRITICAL: Audio must come BEFORE text for Qwen models to transcribe correctly.

        Args:
            message: Message dict that may contain 'audio' or 'audios' fields.

        Returns:
            Message dict with content converted to list format including audio.
        """
        if "audio" not in message and "audios" not in message:
            return message

        content = message.get("content", "")
        if isinstance(content, str):
            message["content"] = [{"type": "text", "text": content}]
        elif isinstance(content, list):
            message["content"] = content
        else:
            raise TypeError(f"Unexpected content type: {type(content)}")

        audio_items = []

        if "audio" in message:
            audio = message["audio"]
            audio_path = os.path.join(self.data_dir, audio["path"])
            base64_audio = audio_file_to_base64(audio_path)
            audio_message = {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{base64_audio}"}}
            audio_items.append(audio_message)
            del message["audio"]  # Remove original audio field after conversion
        elif "audios" in message:
            for audio in message["audios"]:
                audio_path = os.path.join(self.data_dir, audio["path"])
                base64_audio = audio_file_to_base64(audio_path)
                audio_message = {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{base64_audio}"}}
                audio_items.append(audio_message)
            del message["audios"]  # Remove original audios field after conversion

        # Insert audio items at the BEGINNING of content list (before text)
        if audio_items:
            message["content"] = audio_items + message["content"]

        return message

    def _preprocess_messages_for_model(self, messages: list[dict]) -> list[dict]:
        """Preprocess messages - creates copies to avoid mutation.

        Note: /no_think suffix is passed through unchanged (handled by the model).

        Args:
            messages: List of message dicts.

        Returns:
            Copy of message dicts.
        """
        return [copy.deepcopy(msg) for msg in messages]

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
            if msg.get("role") == "user":
                audio_info = msg.get("audio")
                if not audio_info:
                    audios = msg.get("audios", [])
                    audio_info = audios[0] if audios else {}
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
        total_input_tokens = 0
        total_generated_tokens = 0
        total_time = 0.0

        for chunk_idx, audio_chunk in enumerate(chunks):
            chunk_messages = []

            for msg in messages:
                msg_copy = copy.deepcopy(msg)

                if msg_copy["role"] == "user" and ("audio" in msg_copy or "audios" in msg_copy):
                    chunk_base64 = save_audio_chunk_to_base64(audio_chunk, sampling_rate)

                    content = msg_copy.get("content", "")
                    if isinstance(content, str):
                        text_content = [{"type": "text", "text": content}]
                    else:
                        text_content = content

                    # Add audio chunk at the beginning (before text)
                    msg_copy["content"] = [
                        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{chunk_base64}"}}
                    ] + text_content

                    # Remove original audio fields
                    msg_copy.pop("audio", None)
                    msg_copy.pop("audios", None)

                chunk_messages.append(msg_copy)

            # Preprocess messages (strip /no_think for Qwen)
            chunk_messages = self._preprocess_messages_for_model(chunk_messages)

            # Generate for this chunk using parent's generate_async
            result = await super().generate_async(
                prompt=chunk_messages, tokens_to_generate=tokens_to_generate, **kwargs
            )

            # Sum statistics from each chunk
            total_input_tokens += result.get("input_tokens", 0)
            total_generated_tokens += result.get("generated_tokens", 0)
            total_time += result.get("time_elapsed", 0.0)

            generation = result["generation"]
            chunk_results.append(generation.strip())

        # Aggregate results
        aggregated_text = " ".join(chunk_results)

        if not result:
            raise RuntimeError("Audio chunk generation returned no result")

        final_result = result.copy()
        final_result["generation"] = aggregated_text
        final_result["num_audio_chunks"] = len(chunks)
        final_result["audio_duration"] = duration
        # Update with summed statistics
        final_result["input_tokens"] = total_input_tokens
        final_result["generated_tokens"] = total_generated_tokens
        final_result["time_elapsed"] = total_time

        return final_result

    async def generate_async(
        self,
        prompt: str | list[dict] | None = None,
        tokens_to_generate: int | None = None,
        task_type: str = None,
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
            messages = [self.content_text_to_list(copy.deepcopy(msg)) for msg in messages]
            messages = self._preprocess_messages_for_model(messages)
            prompt = messages

        # Call parent's generate_async (which handles audio OUTPUT via _parse_chat_completion_response)
        return await super().generate_async(prompt=prompt, tokens_to_generate=tokens_to_generate, **kwargs)

    def _build_chat_request_params(
        self,
        messages: list[dict],
        **kwargs,
    ) -> dict:
        """Build chat request parameters with audio preprocessing.

        Args:
            messages: List of message dicts.
            **kwargs: Additional parameters for the request.

        Returns:
            Request parameters dict.
        """
        # content_text_to_list THEN preprocess
        messages = [self.content_text_to_list(copy.deepcopy(msg)) for msg in messages]
        messages = self._preprocess_messages_for_model(messages)
        return super()._build_chat_request_params(messages=messages, **kwargs)
