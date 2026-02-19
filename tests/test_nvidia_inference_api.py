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

"""Integration tests for NVIDIA Inference API with VLLMMultimodalModel (audio input)."""

import asyncio
import os
from pathlib import Path

import pytest

from nemo_skills.inference.model.vllm_multimodal import VLLMMultimodalModel

NVIDIA_BASE_URL = "https://inference-api.nvidia.com/v1"
MODEL = "gcp/google/gemini-2.5-flash-lite"

TEST_AUDIO_DIR = Path(__file__).parent / "slurm-tests" / "asr_nim" / "wavs"
TEST_AUDIO_T2 = TEST_AUDIO_DIR / "t2_16.wav"  # "sample 2 this is a test of text to speech synthesis"
TEST_AUDIO_T3 = TEST_AUDIO_DIR / "t3_16.wav"  # "sample 3 hello how are you today"

requires_nvidia_api_key = pytest.mark.skipif(
    not (os.getenv("NV_INFERENCE_API_KEY") or os.getenv("NVIDIA_API_KEY")),
    reason="NV_INFERENCE_API_KEY/NVIDIA_API_KEY environment variable not set",
)

requires_test_audio = pytest.mark.skipif(
    not TEST_AUDIO_T2.exists() or not TEST_AUDIO_T3.exists(),
    reason="Test audio files not found at tests/slurm-tests/asr_nim/wavs/",
)


@requires_nvidia_api_key
def test_nvidia_api_text_only():
    """Smoke test: text-only chat completion via NVIDIA Inference API."""
    model = VLLMMultimodalModel(
        model=MODEL,
        base_url=NVIDIA_BASE_URL,
        audio_format="input_audio",
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Can you help me?"},
    ]

    result = asyncio.run(
        model.generate_async(
            prompt=messages,
            tokens_to_generate=1024,
            temperature=0.7,
        )
    )

    assert "generation" in result
    assert len(result["generation"]) > 0
    print(f"[text-only] generation: {result['generation'][:200]}")


@requires_nvidia_api_key
@requires_test_audio
def test_nvidia_api_audio_input():
    """Integration test: audio-input generation using a local test audio file."""
    model = VLLMMultimodalModel(
        model=MODEL,
        base_url=NVIDIA_BASE_URL,
        audio_format="input_audio",
        data_dir=str(TEST_AUDIO_DIR),
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "What do you hear in this audio? Describe it briefly.",
            "audio": {"path": "t2_16.wav"},
        },
    ]

    result = asyncio.run(
        model.generate_async(
            prompt=messages,
            tokens_to_generate=1024,
            temperature=0.7,
        )
    )

    assert "generation" in result
    assert len(result["generation"]) > 0
    print(f"[audio-input] generation: {result['generation'][:300]}")


@requires_nvidia_api_key
@requires_test_audio
def test_nvidia_api_audio_with_transcription_prompt():
    """Integration test: ask the model to transcribe audio content."""
    model = VLLMMultimodalModel(
        model=MODEL,
        base_url=NVIDIA_BASE_URL,
        audio_format="input_audio",
        data_dir=str(TEST_AUDIO_DIR),
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant that can listen to audio."},
        {
            "role": "user",
            "content": "Please listen to this audio and tell me what you hear.",
            "audio": {"path": "t3_16.wav"},
        },
    ]

    result = asyncio.run(
        model.generate_async(
            prompt=messages,
            tokens_to_generate=1024,
            temperature=0.7,
        )
    )

    assert "generation" in result
    assert len(result["generation"]) > 0
    assert result["num_generated_tokens"] > 0
    print(f"[transcription] generation: {result['generation'][:300]}")
    print(f"[transcription] tokens: {result['num_generated_tokens']}")
