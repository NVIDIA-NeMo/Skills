# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""SALM (Speech-Augmented Language Model) backend for unified_server.

This backend performs speech-to-text using NeMo SALM models (e.g. canary-qwen-2.5b).
Unlike the nemo_asr backend which uses ASRModel.transcribe(), SALM models use a
chat-style generate() API with audio file references embedded in prompts.
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
    """Unified-server backend for SALM models (e.g. canary-qwen-2.5b)."""

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
            raise ValueError("SALM backend requires model_path (or model_name).")

        from nemo.collections.speechlm2.models import SALM

        model_ref = self._model_name

        if Path(model_ref).exists():
            self._model = SALM.restore_from(model_ref)
        else:
            self._model = SALM.from_pretrained(model_ref)

        self._model.to(self.config.device)
        self._model.eval()

        if self.salm_config.warmup:
            self._run_warmup()

        self._is_loaded = True
        logger.info("Loaded SALM model=%s on device=%s", self._model_name, self.config.device)

    def _run_warmup(self) -> None:
        """Run one short warmup call so runtime init happens before traffic."""
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="salm_warmup_")
        os.close(fd)
        try:
            with wave.open(path, "wb") as wavf:
                wavf.setnchannels(1)
                wavf.setsampwidth(2)
                wavf.setframerate(16000)
                wavf.writeframes(b"\x00\x00" * 16000)  # 1 sec silence
            prompts = [
                [
                    {
                        "role": "user",
                        "content": f"{self.salm_config.user_prompt} {self._model.audio_locator_tag}",
                        "audio": [path],
                    }
                ]
            ]
            output_ids = self._model.generate(prompts=prompts, max_new_tokens=8)
            logger.info("SALM warmup completed, output length=%d tokens", len(output_ids[0]))
        except Exception as e:
            logger.warning("SALM warmup skipped due to error: %s", e)
        finally:
            Path(path).unlink(missing_ok=True)

    def _get_request_audio_bytes(self, request: GenerationRequest) -> bytes:
        if request.audio_bytes:
            return request.audio_bytes
        if request.audio_bytes_list:
            if len(request.audio_bytes_list) > 1:
                raise ValueError("SALM backend currently supports one audio input per request.")
            return request.audio_bytes_list[0]
        raise ValueError("Request must contain audio_bytes/audio_bytes_list")

    def validate_request(self, request: GenerationRequest) -> Optional[str]:
        has_audio = request.audio_bytes is not None or (
            request.audio_bytes_list is not None and len(request.audio_bytes_list) > 0
        )
        if not has_audio:
            return "SALM backend requires audio input"
        return None

    def generate(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        if not self._is_loaded:
            return [GenerationResult(error="Model not loaded", request_id=r.request_id) for r in requests]
        if not requests:
            return []

        tmp_dir = Path(tempfile.mkdtemp(prefix="salm_batch_"))
        start = time.time()
        temp_paths: List[str] = []
        valid_indices: List[int] = []
        results: List[Optional[GenerationResult]] = [None] * len(requests)

        try:
            for idx, req in enumerate(requests):
                try:
                    audio_bytes = self._get_request_audio_bytes(req)
                    p = tmp_dir / f"req_{idx:04d}.wav"
                    p.write_bytes(audio_bytes)
                    temp_paths.append(str(p))
                    valid_indices.append(idx)
                except Exception as e:
                    results[idx] = GenerationResult(error=str(e), request_id=req.request_id)

            if temp_paths:
                first_extra = requests[valid_indices[0]].extra_params or {}
                user_prompt = first_extra.get("user_prompt", self.salm_config.user_prompt)
                max_new_tokens = int(first_extra.get("max_new_tokens", self.config.max_new_tokens))
                audio_tag = self._model.audio_locator_tag

                prompts = []
                for path in temp_paths:
                    prompts.append(
                        [
                            {
                                "role": "user",
                                "content": f"{user_prompt} {audio_tag}",
                                "audio": [path],
                            }
                        ]
                    )

                output_ids = self._model.generate(prompts=prompts, max_new_tokens=max_new_tokens)

                if len(output_ids) != len(temp_paths):
                    raise RuntimeError(
                        f"SALM output size mismatch: got {len(output_ids)} for {len(temp_paths)} inputs."
                    )

                per_req_ms = (time.time() - start) * 1000.0 / max(len(temp_paths), 1)
                for out_idx, ids in enumerate(output_ids):
                    req_idx = valid_indices[out_idx]
                    req = requests[req_idx]
                    text = self._model.tokenizer.ids_to_text(ids.cpu())
                    results[req_idx] = GenerationResult(
                        text=text,
                        request_id=req.request_id,
                        num_tokens_generated=len(ids),
                        generation_time_ms=per_req_ms,
                        debug_info={
                            "backend": "salm",
                            "model": self._model_name,
                            "user_prompt": user_prompt,
                        },
                    )

            return [r if r is not None else GenerationResult(error="Unknown SALM backend error") for r in results]
        except Exception as e:
            return [GenerationResult(error=str(e), request_id=r.request_id) for r in requests]
        finally:
            for p in temp_paths:
                Path(p).unlink(missing_ok=True)
            try:
                tmp_dir.rmdir()
            except OSError:
                pass
