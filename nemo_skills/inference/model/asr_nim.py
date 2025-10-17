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

"""asr_nim.py

An *offline* Automatic Speech Recognition (ASR) model wrapper for Riva / NIM
speech containers.

This is the ASR counterpart of ``tts_nim.py`` – it allows NeMo-Skills to treat
an offline‐transcription endpoint as just another *model* that can be invoked
through the common ``BaseModel`` API.

The implementation is intentionally lightweight.  All heavy lifting (gRPC
communication, protobuf message construction, etc.) is delegated to the
official ``riva.client`` Python package that is already shipped with
``nemo-skills``.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from .nim_utils import ASRExtraConfig, setup_ssh_tunnel, validate_unsupported_params

LOG = logging.getLogger(__name__)


class ASRNIMModel:
    """Offline ASR wrapper around a Riva/NIM container (gRPC).

    Parameters
    ----------
    host : str, default "127.0.0.1"
        Hostname or IP of the NIM container.
    port : str, default "50051"
        gRPC port of the service ( *not* the HTTP REST port).
    base_url : str | None
        Ignored – accepted for API compatibility so that callers can supply the
        same parameter set as for other models.
    use_ssl : bool, default ``False``
        Whether to use a TLS-secured channel.
    ssl_cert : str | None
        PEM-encoded certificate chain for *server* verification.
    metadata : list[tuple] | None
        Additional gRPC metadata e.g. for *Bearer* auth.
    language_code : str, default "en-US"
        Language code for the recogniser.
    max_message_length : int, default ``-1``
        Maximum gRPC message size.  ``-1`` means *unlimited* (subject to server
        limits).
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: str = "50051",
        model: str = "riva-asr",
        *,
        base_url: str | None = None,  # accepted for compatibility
        use_ssl: bool = False,
        ssl_cert: Optional[str] = None,
        metadata: Optional[List[tuple]] = None,
        language_code: str = "en-US",
        max_message_length: int = -1,
        max_workers: int = 64,  # Number of threads for concurrent requests
        ssh_server: str | None = None,
        ssh_key_path: str | None = None,
        **kwargs,
    ) -> None:
        # ``base_url`` compatibility – allow callers to pass something like
        # "http://HOST:PORT" and silently strip the scheme/port to get the host.
        if base_url:
            _url = base_url.replace("http://", "").replace("https://", "")
            if ":" in _url:
                host, _ = _url.split(":", 1)
            else:
                host = _url

        # Import riva here to avoid requiring it at module import time
        import riva.client

        # Store reference for later use
        self.riva_client = riva.client

        # Setup SSH tunnel if needed
        host, port, self._tunnel = setup_ssh_tunnel(host, port, ssh_server, ssh_key_path)

        # Store basic attributes (expected by inference code)
        self.model_name_or_path = model
        self.server_host = host
        self.server_port = port

        self.language_code = language_code

        # Build the authenticated gRPC channel
        auth = riva.client.Auth(
            ssl_cert,
            use_ssl,
            f"{host}:{port}",
            metadata,
            max_message_length=max_message_length,
        )
        self.service = riva.client.ASRService(auth)

        # Add thread pool for async execution to avoid blocking the event loop
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        LOG.info(f"Initialized ASRNIMModel with {max_workers} worker threads for concurrent requests")

    # ------------------------------------------------------------------
    # BaseModel API
    # ------------------------------------------------------------------

    async def generate_async(self, prompt: str, **kwargs):
        """Transcribe audio asynchronously using gRPC with proper async handling.

        Runs the blocking gRPC call in a thread pool to avoid blocking the event loop,
        enabling true concurrent request processing.

        Args:
            prompt: Path to audio file to transcribe
            **kwargs: Generation parameters (most LLM parameters are ignored, use extra_body for ASR options)

        Returns:
            dict: Result with 'generation' key containing transcription
        """
        # Validate and warn about unsupported LLM parameters
        validate_unsupported_params(kwargs, "ASRNIMModel")

        loop = asyncio.get_event_loop()
        # Run the blocking call in a thread pool to avoid blocking the event loop
        return await loop.run_in_executor(self._executor, lambda: self._generate_single(prompt, **kwargs))

    def _generate_single(
        self,
        prompt: str,
        # Unused generation parameters – they are accepted so that the wrapper
        # is API-compatible with ``BaseModel``.
        tokens_to_generate: int = 0,
        temperature: float = 0.0,
        top_p: float = 0.0,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        random_seed: int = 0,
        stop_phrases: List[str] | None = None,
        top_logprobs: int | None = None,
        timeout: int | None = None,
        remove_stop_phrases: bool = True,
        stream: bool = False,
        reasoning_effort: str | None = None,
        tools: list | None = None,
        include_response: bool = False,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Transcribe the *entire* audio file specified by *prompt*.

        Parameters
        ----------
        prompt : str
            Path to a WAV/FLAC/etc. audio file on disk.  The service will read
            the *bytes* from this file and send them in **one** request
            («offline» recognition).
        extra_body : dict, optional
            Additional inference parameters – mirrors the CLI arguments of the
            original ``transcribe_file_offline.py`` script.
        """

        if stream:
            raise ValueError("Streaming ASR is not supported by ASRNIMModel. Use the streaming client instead.")

        # Parse and validate ASR parameters from extra_body
        extra_body = extra_body or {}
        try:
            asr_config = ASRExtraConfig(**extra_body)
        except TypeError as e:
            raise ValueError(f"Invalid ASR parameters in extra_body: {e}") from e

        # ------------------------------------------------------------------
        # Build the RecognitionConfig using validated parameters
        # ------------------------------------------------------------------

        language_code = asr_config.language_code or self.language_code
        max_alternatives = asr_config.max_alternatives
        profanity_filter = asr_config.profanity_filter
        automatic_punctuation = asr_config.automatic_punctuation
        no_verbatim_transcripts = asr_config.no_verbatim_transcripts
        word_time_offsets = asr_config.word_time_offsets

        config = self.riva_client.RecognitionConfig(
            language_code=language_code,
            max_alternatives=max_alternatives,
            profanity_filter=profanity_filter,
            enable_automatic_punctuation=automatic_punctuation,
            verbatim_transcripts=(not no_verbatim_transcripts),
            enable_word_time_offsets=word_time_offsets,
        )

        # Word-boosting, diarisation, endpointing, …
        self.riva_client.add_word_boosting_to_config(
            config,
            asr_config.boosted_lm_words,
            asr_config.boosted_lm_score,
        )
        self.riva_client.add_speaker_diarization_to_config(
            config,
            asr_config.speaker_diarization,
            asr_config.diarization_max_speakers,
        )
        self.riva_client.add_endpoint_parameters_to_config(
            config,
            asr_config.start_history,
            asr_config.start_threshold,
            asr_config.stop_history,
            asr_config.stop_history_eou,
            asr_config.stop_threshold,
            asr_config.stop_threshold_eou,
        )
        self.riva_client.add_custom_configuration_to_config(
            config,
            asr_config.custom_configuration,
        )

        # ------------------------------------------------------------------
        # Read the audio file
        # ------------------------------------------------------------------
        audio_path = Path(prompt).expanduser()
        if not audio_path.is_file():
            raise FileNotFoundError(f"Input audio file not found: {audio_path}")

        with audio_path.open("rb") as fh:
            audio_bytes = fh.read()

        # ------------------------------------------------------------------
        # Perform the offline recognition
        # ------------------------------------------------------------------
        start_time = time.time()
        try:
            response = self.service.offline_recognize(audio_bytes, config)
        except Exception as e:  # noqa: BLE001 – we want to re-wrap *any* error
            LOG.error("ASR generation failed: %s", str(e))
            raise

        generation_time = time.time() - start_time

        # ------------------------------------------------------------------
        # Extract and concatenate all transcripts from all chunks/results
        # ------------------------------------------------------------------
        pred_text = ""
        words = []  # collect (word, start, end) if requested

        if getattr(response, "results", None):
            # Concatenate transcripts from all results/chunks
            transcript_parts = []
            for result in response.results:
                if result.alternatives:
                    transcript_parts.append(result.alternatives[0].transcript)
                    # Collect words from this chunk
                    for w in result.alternatives[0].words:
                        words.append(
                            {
                                "start_time": w.start_time / 1000,  # ms → s
                                "end_time": w.end_time / 1000,  # ms → s
                                "word": w.word,
                                "confidence": w.confidence,
                            }
                        )

            # Join all transcript parts with a space
            pred_text = " ".join(transcript_parts)
        else:
            LOG.warning("ASR response contained no results – returning empty transcript.")

        result: Dict[str, Any] = {
            "pred_text": pred_text,
            "words": words,
            "generation_time": generation_time,
            "audio_file": str(audio_path),
        }

        if include_response:
            result["response"] = response  # protobuf message

        return dict(generation=result)

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=False)
        if hasattr(self, "_tunnel") and self._tunnel:
            try:
                self._tunnel.stop()
            except Exception:
                pass  # Ignore errors during cleanup
