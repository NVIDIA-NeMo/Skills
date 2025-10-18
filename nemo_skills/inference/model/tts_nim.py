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

import asyncio
import json
import logging
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

from .nim_utils import TTSExtraConfig, setup_ssh_tunnel, validate_unsupported_params

LOG = logging.getLogger(__name__)


class TTSNIMModel:
    """Wrapper for TTS NIM container using Riva client.

    This model connects to a TTS NIM container via gRPC and generates speech audio
    from text prompts. The audio files are saved to disk and their paths are returned
    as the "generation" output.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: str = "50051",  # Default gRPC port for NIM
        base_url: str = None,  # Accept for compatibility
        model: str = "riva-tts",
        use_ssl: bool = False,
        ssl_cert: str = None,
        metadata: List[tuple] = None,
        voice: str = "Magpie-Multilingual.EN-US.Sofia",  # Default voice
        language_code: str = "en-US",
        sample_rate_hz: int = 22050,
        output_dir: str = "/tmp/tts_outputs",
        max_message_length: int = -1,
        max_workers: int = 64,  # Number of threads for concurrent requests
        ssh_server: str = None,
        ssh_key_path: str = None,
        **kwargs,
    ):
        # If base_url is provided, extract host from it
        if base_url:
            url = base_url.replace("http://", "").replace("https://", "")
            if ":" in url:
                host, _ = url.split(":", 1)
            else:
                host = url

        # Import riva here to avoid requiring it at module import time
        import riva.client
        from riva.client.proto.riva_audio_pb2 import AudioEncoding

        # Store references for later use
        self.AudioEncoding = AudioEncoding
        self.riva_client = riva.client

        # Setup SSH tunnel if needed
        host, port, self._tunnel = setup_ssh_tunnel(host, port, ssh_server, ssh_key_path)

        # Store basic attributes (expected by inference code)
        self.model_name_or_path = model
        self.server_host = host
        self.server_port = port

        self.voice = voice
        self.language_code = language_code
        self.sample_rate_hz = sample_rate_hz
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Riva client
        auth = riva.client.Auth(
            ssl_cert,  # positional argument
            use_ssl,  # positional argument
            f"{host}:{port}",  # server URI
            metadata,  # metadata as positional argument
            max_message_length=max_message_length,
        )
        self.service = riva.client.SpeechSynthesisService(auth)

        # Add thread pool for async execution to avoid blocking the event loop
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        LOG.info(f"Initialized TTSNIMModel with {max_workers} worker threads for concurrent requests")

    def _get_available_voices(self):
        """Fetch and log available voices from the TTS service."""
        try:
            config_response = self.service.stub.GetRivaSynthesisConfig(
                self.riva_client.proto.riva_tts_pb2.RivaSynthesisConfigRequest()
            )

            tts_models = {}
            for model_config in config_response.model_config:
                language_code = model_config.parameters["language_code"]
                voice_name = model_config.parameters["voice_name"]
                subvoices = [voice.split(":")[0] for voice in model_config.parameters["subvoices"].split(",")]
                full_voice_names = [voice_name + "." + subvoice for subvoice in subvoices]

                if language_code in tts_models:
                    tts_models[language_code]["voices"].extend(full_voice_names)
                else:
                    tts_models[language_code] = {"voices": full_voice_names}

            LOG.info("Available TTS voices: %s", json.dumps(tts_models, indent=2))
            self.available_voices = tts_models
        except Exception as e:
            LOG.error(f"Failed to get available voices: {e}")
            self.available_voices = {}

    def _generate_audio_filename(self, text: str, voice: str, idx: int) -> Path:
        """Generate a unique filename for the audio output."""
        # Create a safe filename from text (first 30 chars) and timestamp
        safe_text = "".join(c for c in text[:30] if c.isalnum() or c in (" ", "-", "_")).rstrip()
        safe_text = safe_text.replace(" ", "_")
        timestamp = int(time.time() * 1000)

        filename = f"tts_{idx}_{safe_text}_{timestamp}.wav"
        return self.output_dir / filename

    def _save_audio(self, audio_data: bytes, output_file: Path, sample_rate: int) -> float:
        """Save audio data to a WAV file and return duration in seconds."""
        with wave.open(str(output_file), "wb") as out_f:
            out_f.setnchannels(1)  # Mono
            out_f.setsampwidth(2)  # 16-bit
            out_f.setframerate(sample_rate)
            out_f.writeframes(audio_data)

        # Calculate duration
        num_samples = len(audio_data) / 2  # 16-bit samples
        duration = num_samples / sample_rate
        return duration

    async def generate_async(self, prompt: str, **kwargs):
        """Generate speech asynchronously using gRPC with proper async handling.

        Runs the blocking gRPC call in a thread pool to avoid blocking the event loop,
        enabling true concurrent request processing.

        Args:
            prompt: Text to synthesize
            **kwargs: Generation parameters (most LLM parameters are ignored, use extra_body for TTS options)

        Returns:
            dict: Result with 'generation' key containing audio file path
        """
        # Validate and warn about unsupported LLM parameters
        validate_unsupported_params(kwargs, "TTSNIMModel")

        loop = asyncio.get_event_loop()
        # Run the blocking call in a thread pool to avoid blocking the event loop
        return await loop.run_in_executor(self._executor, lambda: self._generate_single(prompt, **kwargs))

    def _generate_single(
        self,
        prompt: str,
        # Standard parameters from BaseModel
        tokens_to_generate: int = 2048,
        temperature: float = 0.0,
        top_p: float = 0.95,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: List[str] = None,
        top_logprobs: int = None,
        timeout: int = None,
        remove_stop_phrases: bool = True,
        stream: bool = False,
        reasoning_effort: str = None,
        tools: list = None,
        include_response: bool = False,
        extra_body: dict = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Generate speech from a single text prompt.

        Returns a dictionary with 'generation' containing the output audio file path.
        """
        # Parse and validate TTS parameters from extra_body
        extra_body = extra_body or {}
        try:
            tts_config = TTSExtraConfig(**extra_body)
        except TypeError as e:
            raise ValueError(f"Invalid TTS parameters in extra_body: {e}") from e

        # Extract parameters with fallback to instance defaults
        voice = tts_config.voice or self.voice
        language_code = tts_config.language_code or self.language_code
        sample_rate_hz = tts_config.sample_rate_hz or self.sample_rate_hz
        zero_shot_audio_prompt_file = tts_config.zero_shot_audio_prompt_file
        zero_shot_quality = tts_config.zero_shot_quality
        zero_shot_transcript = tts_config.zero_shot_transcript
        custom_dictionary = tts_config.custom_dictionary

        # Validate voice if available voices were fetched
        if hasattr(self, "available_voices") and self.available_voices:
            if language_code in self.available_voices:
                available = self.available_voices[language_code]["voices"]
                if voice not in available:
                    LOG.warning(f"Voice '{voice}' not found. Available voices for {language_code}: {available}")
                    LOG.warning(f"Using default voice: {self.voice}")

        # Convert zero_shot_audio_prompt_file to Path if provided
        if zero_shot_audio_prompt_file:
            zero_shot_audio_prompt_file = Path(zero_shot_audio_prompt_file)

        try:
            # Check if custom output filename is provided
            if tts_config.output_filename:
                # Use custom output filename
                output_file = Path(tts_config.output_filename)
                # Create parent directories if they don't exist
                output_file.parent.mkdir(parents=True, exist_ok=True)
            else:
                # Generate filename in the instance's output directory
                safe_text = "".join(c for c in prompt[:30] if c.isalnum() or c in (" ", "-", "_")).rstrip()
                safe_text = safe_text.replace(" ", "_")
                timestamp = int(time.time() * 1000)
                filename = f"tts_0_{safe_text}_{timestamp}.wav"
                output_file = self.output_dir / filename
            start_time = time.time()

            LOG.debug(f"Synthesizing text: {prompt[:50]}...")

            LOG.info(
                f"Arguments to synthesize: {voice}, {language_code}, {sample_rate_hz}, {zero_shot_audio_prompt_file}, {zero_shot_quality}, {custom_dictionary}, {zero_shot_transcript}"
            )
            resp = self.service.synthesize(
                prompt,
                voice,
                language_code,
                sample_rate_hz=sample_rate_hz,
                encoding=self.AudioEncoding.LINEAR_PCM,
                zero_shot_audio_prompt_file=zero_shot_audio_prompt_file,
                zero_shot_quality=zero_shot_quality,
                custom_dictionary=custom_dictionary or {},
                zero_shot_transcript=zero_shot_transcript,
            )

            duration = self._save_audio(resp.audio, output_file, sample_rate_hz)

            generation_time = time.time() - start_time

            result = {
                "generation": str(output_file),
                "num_generated_tokens": len(resp.audio),  # Audio bytes as proxy
                "generation_time": generation_time,
                "duration": duration,  # Audio duration in seconds
                "text": prompt,
                "voice": voice,
                "language_code": language_code,
                "sample_rate_hz": sample_rate_hz,
            }

            LOG.info(f"Generated audio saved to: {output_file} (took {generation_time:.2f}s)")

            return result

        except Exception as e:
            error_msg = str(e)
            if hasattr(e, "details"):
                error_msg = e.details()
            self._get_available_voices()
            LOG.error(f"TTS generation failed: {error_msg}")

            # Re-raise the exception so BaseModel can handle it properly
            # This prevents the "Generation id not found" error
            raise Exception(f"TTS generation failed: {error_msg}") from e

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
        if hasattr(self, "_tunnel") and self._tunnel:
            try:
                self._tunnel.stop()
            except Exception:
                pass  # Ignore errors during cleanup
