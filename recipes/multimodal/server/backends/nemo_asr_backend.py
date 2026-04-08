# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""NeMo ASR backend for unified_server.

This backend performs offline transcription in batches using NeMo ASR models.
It expects audio input and returns transcript text plus optional word metadata
in debug_info for downstream mapping.
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


@dataclass
class NeMoASRConfig(BackendConfig):
    """Configuration for NeMo ASR backend."""

    # Optional alias for model_path when using --model_name style setup.
    model_name: Optional[str] = None

    # Runtime and batching.
    batch_size: int = 16
    num_workers: int = 0
    return_hypotheses: bool = True
    warmup: bool = True

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "NeMoASRConfig":
        # Allow --model_name to override empty model_path.
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
            "num_workers",
            "return_hypotheses",
            "warmup",
        }
        return cls(
            **{k: v for k, v in d.items() if k in known},
            extra_config={k: v for k, v in d.items() if k not in known},
        )


class NeMoASRBackend(InferenceBackend):
    """Unified-server backend for offline ASR using NeMo models."""

    @classmethod
    def get_config_class(cls) -> type:
        return NeMoASRConfig

    @property
    def name(self) -> str:
        return "nemo_asr"

    @property
    def supported_modalities(self) -> Set[Modality]:
        return {Modality.AUDIO_IN, Modality.TEXT}

    def __init__(self, config: BackendConfig):
        self.asr_config = config if isinstance(config, NeMoASRConfig) else NeMoASRConfig.from_dict(config.extra_config)
        super().__init__(self.asr_config)
        self._model_name = self.asr_config.model_path or self.asr_config.model_name
        self._model = None

    def load_model(self) -> None:
        if not self._model_name:
            raise ValueError("NeMo ASR backend requires model_path (or model_name).")

        import torch
        from nemo.collections.asr.models import ASRModel

        model_ref = self._model_name
        map_location = torch.device(self.config.device)

        if Path(model_ref).exists():
            self._model = ASRModel.restore_from(model_ref, map_location=map_location)
        else:
            self._model = ASRModel.from_pretrained(model_name=model_ref, map_location=map_location)

        if not hasattr(self._model, "to"):
            raise RuntimeError(f"Loaded ASR model '{self._model_name}' does not support `.to(device)` placement.")
        try:
            self._model.to(self.config.device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to move ASR model '{self._model_name}' to device '{self.config.device}'."
            ) from e
        self._model.eval()

        if self.asr_config.warmup:
            self._run_warmup()

        self._is_loaded = True
        logger.info("Loaded NeMo ASR model=%s on device=%s", self._model_name, self.config.device)

    def _run_warmup(self) -> None:
        """Run one short warmup call so runtime init happens before traffic."""
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="nemo_asr_warmup_")
        os.close(fd)
        try:
            with wave.open(path, "wb") as wavf:
                wavf.setnchannels(1)
                wavf.setsampwidth(2)
                wavf.setframerate(16000)
                wavf.writeframes(b"\x00\x00" * 1600)  # 0.1 sec silence
            self._transcribe_paths([path], return_hypotheses=False, batch_size=1)
        except Exception as e:
            logger.warning("ASR warmup skipped due to error: %s", e)
        finally:
            Path(path).unlink(missing_ok=True)

    def _transcribe_paths(
        self,
        audio_paths: List[str],
        *,
        return_hypotheses: bool,
        batch_size: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Call NeMo transcribe with compatibility across signatures."""
        kwargs = {
            "batch_size": batch_size,
            "return_hypotheses": return_hypotheses,
            "num_workers": self.asr_config.num_workers,
        }
        if extra:
            kwargs.update(extra)

        try:
            return self._model.transcribe(audio=audio_paths, **kwargs)
        except TypeError:
            try:
                return self._model.transcribe(paths2audio_files=audio_paths, **kwargs)
            except TypeError:
                return self._model.transcribe(audio_paths, **kwargs)

    @staticmethod
    def _normalize_words(words_obj: Any) -> List[Dict[str, Any]]:
        """Normalize various word/timestamp schemas to list[dict]."""
        if words_obj is None:
            return []

        if isinstance(words_obj, list):
            normalized = []
            for item in words_obj:
                if isinstance(item, dict):
                    normalized.append(
                        {
                            "word": item.get("word", item.get("text", "")),
                            "start_time": item.get("start_time", item.get("start", None)),
                            "end_time": item.get("end_time", item.get("end", None)),
                            "confidence": item.get("confidence", None),
                        }
                    )
                    continue
                if isinstance(item, (tuple, list)):
                    word = item[0] if len(item) > 0 else ""
                    start = item[1] if len(item) > 1 else None
                    end = item[2] if len(item) > 2 else None
                    normalized.append({"word": word, "start_time": start, "end_time": end, "confidence": None})
                    continue
                if isinstance(item, str):
                    normalized.append({"word": item, "start_time": None, "end_time": None, "confidence": None})
                    continue

                word = getattr(item, "word", getattr(item, "text", ""))
                normalized.append(
                    {
                        "word": word,
                        "start_time": getattr(item, "start_time", getattr(item, "start", None)),
                        "end_time": getattr(item, "end_time", getattr(item, "end", None)),
                        "confidence": getattr(item, "confidence", None),
                    }
                )
            return normalized

        if isinstance(words_obj, dict):
            words = words_obj.get("word") or words_obj.get("words")
            starts = words_obj.get("start") or words_obj.get("start_time")
            ends = words_obj.get("end") or words_obj.get("end_time")
            if isinstance(words, list):
                out = []
                for idx, word in enumerate(words):
                    start = starts[idx] if isinstance(starts, list) and idx < len(starts) else None
                    end = ends[idx] if isinstance(ends, list) and idx < len(ends) else None
                    out.append({"word": word, "start_time": start, "end_time": end, "confidence": None})
                return out

        return []

    def _parse_single_hypothesis(self, hyp: Any) -> tuple[str, List[Dict[str, Any]]]:
        """Extract transcript and words from heterogeneous NeMo outputs."""
        if isinstance(hyp, str):
            return hyp, []

        if isinstance(hyp, dict):
            text = hyp.get("text")
            if text is None:
                text = hyp.get("pred_text")
            if text is None:
                text = hyp.get("transcript")
            if text is None:
                text = ""
            words = hyp.get("words")
            if words is None:
                ts = hyp.get("timestamp")
                if isinstance(ts, dict):
                    words = ts.get("word")
            elif isinstance(words, list) and all(isinstance(w, str) for w in words):
                ts = hyp.get("timestamp")
                if isinstance(ts, dict) and isinstance(ts.get("word"), list):
                    words = ts["word"]
            return text, self._normalize_words(words)

        text = getattr(hyp, "text", None)
        if text is None:
            text = getattr(hyp, "pred_text", None)
        if text is None:
            text = ""
        words = getattr(hyp, "words", None)
        if words is None:
            ts = getattr(hyp, "timestamp", None)
            if isinstance(ts, dict):
                words = ts.get("word")
        elif isinstance(words, list) and all(isinstance(w, str) for w in words):
            ts = getattr(hyp, "timestamp", None)
            if isinstance(ts, dict) and isinstance(ts.get("word"), list):
                words = ts["word"]
        return text, self._normalize_words(words)

    def _get_request_audio_bytes(self, request: GenerationRequest) -> bytes:
        if request.audio_bytes:
            return request.audio_bytes
        if request.audio_bytes_list:
            if len(request.audio_bytes_list) > 1:
                raise ValueError("nemo_asr backend currently supports one audio input per request.")
            return request.audio_bytes_list[0]
        raise ValueError("Request must contain audio_bytes/audio_bytes_list")

    def validate_request(self, request: GenerationRequest) -> Optional[str]:
        has_audio = request.audio_bytes is not None or (
            request.audio_bytes_list is not None and len(request.audio_bytes_list) > 0
        )
        if not has_audio:
            return "nemo_asr backend requires audio input"
        return None

    def generate(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        if not self._is_loaded:
            return [GenerationResult(error="Model not loaded", request_id=r.request_id) for r in requests]
        if not requests:
            return []

        tmp_dir = Path(tempfile.mkdtemp(prefix="nemo_asr_batch_"))
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
                return_hypotheses = bool(first_extra.get("return_hypotheses", self.asr_config.return_hypotheses))
                transcribe_batch_size = int(first_extra.get("batch_size", self.asr_config.batch_size))
                transcribe_batch_size = max(1, min(transcribe_batch_size, len(temp_paths)))

                # Pass through optional ASR params if present (useful for canary-style models).
                optional_keys = ["timestamps", "task", "source_lang", "target_lang", "pnc", "channel_selector"]
                extra = {k: first_extra[k] for k in optional_keys if k in first_extra}

                hyps = self._transcribe_paths(
                    temp_paths,
                    return_hypotheses=return_hypotheses,
                    batch_size=transcribe_batch_size,
                    extra=extra,
                )

                if not isinstance(hyps, list):
                    hyps = [hyps]
                if len(hyps) != len(temp_paths):
                    raise RuntimeError(f"ASR output size mismatch: got {len(hyps)} for {len(temp_paths)} inputs.")

                per_req_ms = (time.time() - start) * 1000.0 / max(len(temp_paths), 1)
                for out_idx, hyp in enumerate(hyps):
                    req_idx = valid_indices[out_idx]
                    req = requests[req_idx]
                    text, words = self._parse_single_hypothesis(hyp)
                    results[req_idx] = GenerationResult(
                        text=text,
                        request_id=req.request_id,
                        generation_time_ms=per_req_ms,
                        debug_info={
                            "words": words,
                            "backend": "nemo_asr",
                            "model": self._model_name,
                            "batch_size": transcribe_batch_size,
                        },
                    )

            return [r if r is not None else GenerationResult(error="Unknown ASR backend error") for r in results]
        except Exception as e:
            return [GenerationResult(error=str(e), request_id=r.request_id) for r in requests]
        finally:
            for p in temp_paths:
                Path(p).unlink(missing_ok=True)
            try:
                tmp_dir.rmdir()
            except OSError:
                pass
