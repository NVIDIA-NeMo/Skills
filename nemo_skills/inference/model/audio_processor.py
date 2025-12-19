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

"""Audio processing wrapper for multimodal models.

This module provides an AudioProcessor wrapper that can be composed with any
BaseModel to add audio preprocessing capabilities. It handles:
- Converting audio file paths to base64-encoded audio_url format
- Chunking long audio files for models with duration limits
- Aggregating results from chunked audio processing
"""

import base64
import logging
import os

from nemo_skills.utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class AudioProcessorConfig:
    """Configuration for audio preprocessing.

    Attributes:
        data_dir: Base directory for resolving relative audio file paths.
        enable_chunking: Whether to chunk long audio files.
        chunk_task_types: If None, chunk all task types; if specified, only chunk these.
        chunk_threshold_sec: Audio duration threshold (in seconds) above which to chunk.
    """

    data_dir: str = ""
    enable_chunking: bool = True
    chunk_task_types: list[str] | None = None
    chunk_threshold_sec: int = 30


def audio_file_to_base64(audio_file_path: str) -> str:
    """Encodes an audio file into a base64 string."""
    with open(audio_file_path, "rb") as audio_file:
        audio_content = audio_file.read()
        return base64.b64encode(audio_content).decode("utf-8")


def load_audio_file(audio_file_path: str):
    """Load audio file and return array and sampling rate."""
    import soundfile as sf

    audio_array, sampling_rate = sf.read(audio_file_path)
    return audio_array, sampling_rate


def chunk_audio(audio_array, sampling_rate, chunk_duration_sec=30):
    """Chunk audio array into segments of specified duration.

    Args:
        audio_array: Audio data as numpy array
        sampling_rate: Sampling rate in Hz
        chunk_duration_sec: Duration of each chunk in seconds

    Returns:
        List of audio chunks
    """
    import numpy as np

    chunk_samples = int(chunk_duration_sec * sampling_rate)
    num_chunks = int(np.ceil(len(audio_array) / chunk_samples))

    chunks = []
    for i in range(num_chunks):
        start = i * chunk_samples
        end = min((i + 1) * chunk_samples, len(audio_array))
        chunks.append(audio_array[start:end])

    return chunks


def save_audio_chunk_to_base64(audio_chunk, sampling_rate) -> str:
    """Save audio chunk to temporary file and convert to base64.

    Args:
        audio_chunk: Audio data as numpy array
        sampling_rate: Sampling rate in Hz

    Returns:
        Base64 encoded audio string
    """
    import tempfile

    import soundfile as sf

    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        sf.write(tmp_path, audio_chunk, sampling_rate)

    try:
        # Read and encode
        with open(tmp_path, "rb") as f:
            audio_content = f.read()
            encoded = base64.b64encode(audio_content).decode("utf-8")
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return encoded


class AudioProcessor:
    """Wraps any model to add audio preprocessing capabilities.

    This wrapper handles:
    - Converting audio file paths in messages to base64-encoded audio_url format
    - Chunking long audio files and aggregating results
    - Passing through all other requests unchanged

    Example usage:
        model = get_model(server_type="vllm", ...)
        audio_model = AudioProcessor(model, AudioProcessorConfig(), eval_config={...}, eval_type="audio")
        result = await audio_model.generate_async(prompt=messages, ...)
    """

    def __init__(
        self,
        model,
        config: AudioProcessorConfig,
        eval_config: dict | None = None,
        eval_type: str | None = None,
    ):
        """Initialize AudioProcessor wrapper.

        Args:
            model: The underlying model to wrap (must have generate_async method)
            config: Audio processing configuration
            eval_config: Optional eval config dict (contains "data_dir" key) for inferring data_dir
            eval_type: Optional eval type string for inferring data_dir
        """
        self.model = model
        self.config = config

        # Resolve data_dir: explicit config takes precedence, then infer from eval_config
        if config.data_dir:
            self.data_dir = config.data_dir
        elif eval_config is not None and eval_type is not None:
            eval_data_dir = eval_config.get("data_dir")
            if eval_data_dir is not None:
                self.data_dir = os.path.join(eval_data_dir, eval_type)
            else:
                self.data_dir = ""
        else:
            self.data_dir = ""

        # Expose common model attributes for compatibility
        if hasattr(model, "model_name_or_path"):
            self.model_name_or_path = model.model_name_or_path
        if hasattr(model, "tokenizer"):
            self.tokenizer = model.tokenizer

    async def generate_async(
        self,
        prompt: str | list[dict] | None = None,
        task_type: str = None,
        **kwargs,
    ) -> dict:
        """Generate with automatic audio preprocessing and chunking.

        If the prompt contains audio that needs chunking, processes each chunk
        separately and aggregates results. Otherwise, converts audio to base64
        and passes through to the underlying model.

        Args:
            prompt: Either a string (text completion) or list of messages (chat)
            task_type: Optional task type for chunking filtering
            **kwargs: Additional arguments passed to the underlying model

        Returns:
            Generation result dict with 'generation' key and optional metadata
        """
        if isinstance(prompt, list):
            messages = prompt
            needs_chunking, audio_path, duration = self._check_chunking_needed(messages, task_type)

            if needs_chunking:
                return await self._generate_with_chunking(messages, audio_path, duration, **kwargs)

            # Convert audio fields to base64 format
            messages = self._prepare_audio_messages(messages)
            prompt = messages

        return await self.model.generate_async(prompt=prompt, **kwargs)

    def _prepare_audio_messages(self, messages: list[dict]) -> list[dict]:
        """Convert audio file references in messages to base64-encoded audio_url format.

        Handles 'audio' or 'audios' keys in messages and converts them to
        base64-encoded audio_url content items.

        CRITICAL: Audio must come BEFORE text for Qwen models to transcribe correctly.
        """
        prepared_messages = []

        for message in messages:
            msg = message.copy()

            if "audio" not in msg and "audios" not in msg:
                prepared_messages.append(msg)
                continue

            # Convert content to list format if needed
            content = msg.get("content", "")
            if isinstance(content, str):
                text_content = [{"type": "text", "text": content}]
            elif isinstance(content, list):
                text_content = content
            else:
                raise TypeError(f"Unexpected content type: {type(content)}")

            # Build audio content items
            audio_items = []

            if "audio" in msg:
                audio = msg["audio"]
                audio_path = os.path.join(self.data_dir, audio["path"])
                base64_audio = audio_file_to_base64(audio_path)
                audio_items.append(
                    {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{base64_audio}"}}
                )
                del msg["audio"]
            elif "audios" in msg:
                for audio in msg["audios"]:
                    audio_path = os.path.join(self.data_dir, audio["path"])
                    base64_audio = audio_file_to_base64(audio_path)
                    audio_items.append(
                        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{base64_audio}"}}
                    )
                del msg["audios"]

            # Audio items BEFORE text content (required for Qwen models)
            msg["content"] = audio_items + text_content
            prepared_messages.append(msg)

        return prepared_messages

    def _check_chunking_needed(self, messages: list[dict], task_type: str = None) -> tuple[bool, str, float]:
        """Check if audio in messages needs chunking.

        Returns:
            Tuple of (needs_chunking, audio_path, duration)
        """
        if not self.config.enable_chunking:
            return False, None, 0.0

        # Check if task type should be chunked (if filter is specified)
        if self.config.chunk_task_types is not None:
            if task_type not in self.config.chunk_task_types:
                return False, None, 0.0

        # Find audio in messages
        for msg in messages:
            if msg.get("role") == "user":
                audio_info = msg.get("audio") or (msg.get("audios", [{}])[0] if msg.get("audios") else {})
                if audio_info and "path" in audio_info:
                    audio_path = os.path.join(self.data_dir, audio_info["path"])

                    if not os.path.exists(audio_path):
                        return False, None, 0.0

                    # Load audio to check duration
                    try:
                        audio_array, sampling_rate = load_audio_file(audio_path)
                        duration = len(audio_array) / sampling_rate

                        if duration > self.config.chunk_threshold_sec:
                            return True, audio_path, duration
                    except Exception:
                        pass

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
            messages: Original messages containing audio reference
            audio_path: Path to the audio file to chunk
            duration: Duration of audio in seconds
            tokens_to_generate: Max tokens per chunk
            **kwargs: Additional generation parameters

        Returns:
            Aggregated result with combined generation from all chunks
        """
        audio_array, sampling_rate = load_audio_file(audio_path)
        chunks = chunk_audio(audio_array, sampling_rate, self.config.chunk_threshold_sec)

        LOG.info(f"Chunking audio ({duration:.1f}s) into {len(chunks)} chunks of {self.config.chunk_threshold_sec}s")

        chunk_results = []
        result = None

        for chunk_idx, audio_chunk in enumerate(chunks):
            chunk_messages = []

            for msg in messages:
                msg_copy = msg.copy()

                if msg_copy.get("role") == "user" and ("audio" in msg_copy or "audios" in msg_copy):
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

            result = await self.model.generate_async(
                prompt=chunk_messages, tokens_to_generate=tokens_to_generate, **kwargs
            )

            generation = result.get("generation", "")
            chunk_results.append(generation.strip())

        # Aggregate results
        aggregated_text = " ".join(chunk_results)

        if result:
            final_result = result.copy()
            final_result["generation"] = aggregated_text
            final_result["num_audio_chunks"] = len(chunks)
            final_result["audio_duration"] = duration
        else:
            final_result = {
                "generation": aggregated_text,
                "num_audio_chunks": len(chunks),
                "audio_duration": duration,
            }

        return final_result

    # Proxy other common methods to the underlying model
    def __getattr__(self, name):
        """Proxy attribute access to the underlying model."""
        return getattr(self.model, name)
