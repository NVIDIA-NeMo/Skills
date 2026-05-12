# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0

"""SALM backend for unified_server.

This supports generic Canary-Qwen style ASR through NeMo SALM.generate().
It is not a LibriSpeechMix SOT-specific backend and does not add speaker
supervision by itself.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .base import BackendConfig, GenerationRequest, GenerationResult, InferenceBackend, Modality

logger = logging.getLogger(__name__)

DEFAULT_ASR_PROMPT = "Transcribe the following:"


@dataclass
class SALMConfig(BackendConfig):
    """Configuration for SALM backend."""

    model_name: Optional[str] = None
    batch_size: int = 16
    warmup: bool = True
    user_prompt: str = DEFAULT_ASR_PROMPT

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SALMConfig":
        if d.get("model_name") and not d.get("model_path"):
            d = {**d, "model_path": d["model_name"]}
        known = {
            "model_path",
            "model_name",
            "device",
            "dtype",
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "batch_size",
            "warmup",
            "user_prompt",
        }
        return cls(
            **{k: v for k, v in d.items() if k in known},
            extra_config={k: v for k, v in d.items() if k not in known},
        )


class SALMBackend(InferenceBackend):
    """Unified-server backend for SALM models such as nvidia/canary-qwen-2.5b."""

    @classmethod
    def get_config_class(cls) -> type:
        return SALMConfig

    @property
    def name(self) -> str:
        return "salm"

    @property
    def supported_modalities(self) -> Set[Modality]:
        return {Modality.AUDIO_IN, Modality.TEXT}

    def __init__(self, config: BackendConfig):
        self.salm_config = config if isinstance(config, SALMConfig) else SALMConfig.from_dict(config.extra_config)
        super().__init__(self.salm_config)
        self._model_name = self.salm_config.model_path or self.salm_config.model_name
        self._model = None

    def load_model(self) -> None:
        if not self._model_name:
            raise ValueError("SALM backend requires model_path or model_name.")

        from nemo.collections.speechlm2.models import SALM

        if Path(self._model_name).exists():
            self._model = SALM.restore_from(self._model_name)
        else:
            self._model = SALM.from_pretrained(self._model_name)

        self._model.to(self.config.device)
        self._model.eval()

        if self.salm_config.warmup:
            self._run_warmup()

        self._is_loaded = True
        logger.info("Loaded SALM model=%s on device=%s", self._model_name, self.config.device)

    def _run_warmup(self) -> None:
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="salm_warmup_")
        os.close(fd)
        try:
            with wave.open(path, "wb") as wavf:
                wavf.setnchannels(1)
                wavf.setsampwidth(2)
                wavf.setframerate(16000)
                wavf.writeframes(b"\x00\x00" * 16000)
            prompts = [
                [
                    {
                        "role": "user",
                        "content": f"{self.salm_config.user_prompt} {self._model.audio_locator_tag}",
                        "audio": [path],
                    }
                ]
            ]
            self._model.generate(prompts=prompts, max_new_tokens=8)
        except Exception as exc:
            logger.warning("SALM warmup skipped due to error: %s", exc)
        finally:
            Path(path).unlink(missing_ok=True)

    def _get_request_audio_bytes(self, request: GenerationRequest) -> bytes:
        if request.audio_bytes:
            return request.audio_bytes
        if request.audio_bytes_list:
            if len(request.audio_bytes_list) > 1:
                raise ValueError("SALM backend currently supports one audio input per request.")
            return request.audio_bytes_list[0]
        raise ValueError("Request must contain audio_bytes or audio_bytes_list.")

    def validate_request(self, request: GenerationRequest) -> Optional[str]:
        if request.audio_bytes is None and not request.audio_bytes_list:
            return "SALM backend requires audio input"
        if request.audio_bytes_list is not None and len(request.audio_bytes_list) > 1:
            return "SALM backend currently supports one audio input per request"
        return None

    def generate(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        if not self._is_loaded:
            return [GenerationResult(error="Model not loaded", request_id=request.request_id) for request in requests]
        if not requests:
            return []

        tmp_dir = Path(tempfile.mkdtemp(prefix="salm_batch_"))
        start = time.time()
        temp_paths: list[str] = []
        valid_indices: list[int] = []
        results: list[GenerationResult | None] = [None] * len(requests)

        try:
            for idx, request in enumerate(requests):
                try:
                    path = tmp_dir / f"req_{idx:04d}.wav"
                    path.write_bytes(self._get_request_audio_bytes(request))
                    temp_paths.append(str(path))
                    valid_indices.append(idx)
                except Exception as exc:
                    results[idx] = GenerationResult(error=str(exc), request_id=request.request_id)

            if temp_paths:
                prompts = []
                max_new_tokens = self.config.max_new_tokens
                audio_tag = self._model.audio_locator_tag
                for out_idx, path in enumerate(temp_paths):
                    request = requests[valid_indices[out_idx]]
                    user_prompt = request.user_prompt or request.extra_params.get("user_prompt", self.salm_config.user_prompt)
                    prompts.append([{"role": "user", "content": f"{user_prompt} {audio_tag}", "audio": [path]}])
                    if request.max_new_tokens:
                        max_new_tokens = max(max_new_tokens, request.max_new_tokens)

                output_ids = self._model.generate(prompts=prompts, max_new_tokens=max_new_tokens)
                if len(output_ids) != len(temp_paths):
                    raise RuntimeError(f"SALM output size mismatch: got {len(output_ids)} for {len(temp_paths)} inputs.")

                per_req_ms = (time.time() - start) * 1000.0 / max(len(temp_paths), 1)
                for out_idx, ids in enumerate(output_ids):
                    req_idx = valid_indices[out_idx]
                    request = requests[req_idx]
                    text = self._model.tokenizer.ids_to_text(ids.cpu())
                    results[req_idx] = GenerationResult(
                        text=text,
                        request_id=request.request_id,
                        num_tokens_generated=len(ids),
                        generation_time_ms=per_req_ms,
                        debug_info={"backend": "salm", "model": self._model_name},
                    )
            return [result if result is not None else GenerationResult(error="Unknown SALM backend error") for result in results]
        except Exception as exc:
            return [GenerationResult(error=str(exc), request_id=request.request_id) for request in requests]
        finally:
            for path in temp_paths:
                Path(path).unlink(missing_ok=True)
            try:
                tmp_dir.rmdir()
            except OSError:
                pass
