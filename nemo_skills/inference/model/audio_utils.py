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

"""Shared audio utility functions for multimodal models.

This module provides helper functions for audio processing that can be used
by VLLMMultimodalModel and other audio-capable model classes.
"""

import base64
import logging
import os

from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def audio_file_to_base64(audio_file_path: str) -> str:
    """Encodes an audio file into a base64 string.

    Args:
        audio_file_path: Path to the audio file to encode.

    Returns:
        Base64 encoded string of the audio file contents.
    """
    with open(audio_file_path, "rb") as audio_file:
        audio_content = audio_file.read()
        return base64.b64encode(audio_content).decode("utf-8")


def load_audio_file(audio_file_path: str):
    """Load audio file and return array and sampling rate.

    Args:
        audio_file_path: Path to the audio file to load.

    Returns:
        Tuple of (audio_array, sampling_rate).
    """
    import soundfile as sf

    audio_array, sampling_rate = sf.read(audio_file_path)
    return audio_array, sampling_rate


def chunk_audio(audio_array, sampling_rate, chunk_duration_sec=30, min_chunk_duration_sec=0.5):
    """Chunk audio array into segments of specified duration.

    Args:
        audio_array: Audio data as numpy array.
        sampling_rate: Sampling rate in Hz.
        chunk_duration_sec: Duration of each chunk in seconds.
        min_chunk_duration_sec: Minimum duration for last chunk (shorter chunks are merged).

    Returns:
        List of audio chunks (numpy arrays).
    """
    import numpy as np

    chunk_samples = int(chunk_duration_sec * sampling_rate)
    min_chunk_samples = int(min_chunk_duration_sec * sampling_rate)

    # Validate minimum audio length
    if len(audio_array) < min_chunk_samples:
        raise ValueError(
            f"Audio too short: {len(audio_array) / sampling_rate:.2f}s < minimum {min_chunk_duration_sec}s"
        )

    num_chunks = int(np.ceil(len(audio_array) / chunk_samples))

    chunks = []
    for i in range(num_chunks):
        start = i * chunk_samples
        end = min((i + 1) * chunk_samples, len(audio_array))
        chunk = audio_array[start:end]

        # Merge tiny trailing chunks with previous chunk to avoid empty audio errors
        if len(chunk) < min_chunk_samples and chunks:
            chunks[-1] = np.concatenate([chunks[-1], chunk])
        else:
            chunks.append(chunk)

    return chunks


def save_audio_chunk_to_base64(audio_chunk, sampling_rate) -> str:
    """Save audio chunk to temporary file and convert to base64.

    Args:
        audio_chunk: Audio data as numpy array.
        sampling_rate: Sampling rate in Hz.

    Returns:
        Base64 encoded audio string.
    """
    import tempfile

    import soundfile as sf

    # Create temporary file
    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp_file.name

    try:
        tmp_file.close()
        sf.write(tmp_path, audio_chunk, sampling_rate)

        # Read and encode
        with open(tmp_path, "rb") as f:
            audio_content = f.read()
            encoded = base64.b64encode(audio_content).decode("utf-8")
    finally:
        # Clean up
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    return encoded


def make_audio_content_block(base64_audio: str, audio_format: str = "audio_url") -> dict:
    """Create an audio content block in the specified format.

    Args:
        base64_audio: Base64-encoded audio data.
        audio_format: Format to use:
            - "audio_url": Data URI format for vLLM/Qwen
            - "input_audio": OpenAI native format for NVIDIA API/Gemini/Azure

    Returns:
        Audio content block dict for API request.
    """
    if audio_format == "input_audio":
        # OpenAI native format (works with NVIDIA API / Gemini / Azure)
        return {"type": "input_audio", "input_audio": {"data": base64_audio, "format": "wav"}}
    elif audio_format == "audio_url":
        # Data URI format (works with vLLM / Qwen)
        return {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{base64_audio}"}}
    else:
        raise ValueError(f"Unsupported audio_format '{audio_format}'. Use 'audio_url' or 'input_audio'.")
