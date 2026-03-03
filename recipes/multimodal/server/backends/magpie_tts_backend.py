# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0

"""MagpieTTS backend using MagpieInferenceRunner with RTF metrics."""

import inspect
import io
import json
import logging
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import soundfile as sf

from .base import BackendConfig, GenerationRequest, GenerationResult, InferenceBackend, Modality

logger = logging.getLogger(__name__)

try:
    import nemo.collections.tts.modules.audio_codec_modules as _audio_codec_modules
except ImportError:
    _audio_codec_modules = None

try:
    from nemo.collections.tts.modules.magpietts_inference.evaluate_generated_audio import (
        load_evalset_config as _load_evalset_config,
    )
    from nemo.collections.tts.modules.magpietts_inference.inference import (
        InferenceConfig as _InferenceConfig,
    )
    from nemo.collections.tts.modules.magpietts_inference.inference import (
        MagpieInferenceRunner as _MagpieInferenceRunner,
    )
    from nemo.collections.tts.modules.magpietts_inference.utils import (
        ModelLoadConfig as _ModelLoadConfig,
    )
    from nemo.collections.tts.modules.magpietts_inference.utils import (
        load_magpie_model as _load_magpie_model,
    )
except ImportError:
    _load_evalset_config = None
    _InferenceConfig = None
    _MagpieInferenceRunner = None
    _ModelLoadConfig = None
    _load_magpie_model = None

try:
    from nemo.collections.tts.models.magpietts import ModelInferenceParameters as _ModelInferenceParameters
except ImportError:
    _ModelInferenceParameters = None

try:
    from huggingface_hub import hf_hub_download as _hf_hub_download
except ImportError:
    _hf_hub_download = None

try:
    from nemo_text_processing.text_normalization.normalize import Normalizer as _Normalizer
except ImportError:
    _Normalizer = None


@dataclass
class MagpieTTSConfig(BackendConfig):
    codec_model_path: Optional[str] = None
    max_decoder_steps: Optional[int] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    cfg_scale: Optional[float] = None
    use_cfg: bool = True
    use_local_transformer: bool = True
    apply_attention_prior: bool = True
    output_sample_rate: int = 22050
    # Checkpoint loading options (alternative to model_path .nemo file)
    hparams_file: Optional[str] = None
    checkpoint_file: Optional[str] = None
    legacy_codebooks: bool = False
    legacy_text_conditioning: bool = False
    hparams_from_wandb: bool = False
    # Text normalization (expands numbers, abbreviations, etc. before TTS)
    enable_normalization: bool = False
    normalizer_lang: str = "en"
    normalizer_input_case: str = "cased"
    # Optional allowlist for request-provided context_audio_filepath values.
    context_audio_allowed_roots: Optional[List[str]] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MagpieTTSConfig":
        # Handle CLI alias: --codec_model → codec_model_path
        if "codec_model" in d and "codec_model_path" not in d:
            d = {**d, "codec_model_path": d.pop("codec_model")}
        if isinstance(d.get("context_audio_allowed_roots"), str):
            d = {
                **d,
                "context_audio_allowed_roots": [p for p in d["context_audio_allowed_roots"].split(":") if p],
            }
        known = {
            "model_path",
            "device",
            "dtype",
            "max_new_tokens",
            "temperature",
            "top_p",
            "top_k",
            "codec_model_path",
            "use_cfg",
            "cfg_scale",
            "max_decoder_steps",
            "use_local_transformer",
            "apply_attention_prior",
            "output_sample_rate",
            "hparams_file",
            "checkpoint_file",
            "legacy_codebooks",
            "legacy_text_conditioning",
            "hparams_from_wandb",
            "enable_normalization",
            "normalizer_lang",
            "normalizer_input_case",
            "context_audio_allowed_roots",
        }
        return cls(
            **{k: v for k, v in d.items() if k in known}, extra_config={k: v for k, v in d.items() if k not in known}
        )


class MagpieTTSBackend(InferenceBackend):
    """MagpieTTS backend. Input: JSON with 'text' and 'context_audio_filepath'."""

    @classmethod
    def get_config_class(cls) -> type:
        return MagpieTTSConfig

    @property
    def name(self) -> str:
        return "magpie_tts"

    @property
    def supported_modalities(self) -> Set[Modality]:
        return {Modality.TEXT, Modality.AUDIO_OUT}

    def __init__(self, config: BackendConfig):
        self.tts_config = (
            config
            if isinstance(config, MagpieTTSConfig)
            else MagpieTTSConfig.from_dict(
                {
                    **{
                        k: getattr(config, k)
                        for k in ["model_path", "device", "dtype", "max_new_tokens", "temperature", "top_p", "top_k"]
                        if hasattr(config, k)
                    },
                    **config.extra_config,
                }
            )
        )
        super().__init__(self.tts_config)
        self._model = self._runner = self._temp_dir = self._checkpoint_name = None
        self._normalizer = None

    def _patch_hf_fsspec_loader(self) -> None:
        """Patch NeMo load_fsspec to use hf_hub_download for HF resolve URLs."""
        if _audio_codec_modules is None:
            logger.warning("nemo TTS audio codec modules are unavailable; skipping load_fsspec HF patch")
            return

        orig_load_fsspec = getattr(_audio_codec_modules, "load_fsspec", None)
        if not callable(orig_load_fsspec) or getattr(_audio_codec_modules, "_hf_load_fsspec_patched", False):
            return

        if _hf_hub_download is None:
            logger.warning("huggingface_hub is unavailable; skipping load_fsspec HF patch")
            return

        def _hf_resolve_to_local(url: str) -> str | None:
            if not isinstance(url, str):
                return None
            url_no_q = url.split("?", 1)[0]
            match = re.match(r"^https?://huggingface\.co/([^/]+)/([^/]+)/resolve/([^/]+)/(.+)$", url_no_q)
            if not match:
                return None
            repo_id = f"{match.group(1)}/{match.group(2)}"
            revision = match.group(3)
            filename = match.group(4)
            token = os.environ.get("HF_TOKEN") or None
            return _hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, token=token)

        def _load_fsspec_patched(path: str, map_location: str = None, **kwargs):
            if isinstance(path, str) and path.startswith("http"):
                local_path = _hf_resolve_to_local(path)
                if local_path:
                    return orig_load_fsspec(local_path, map_location=map_location, **kwargs)
            return orig_load_fsspec(path, map_location=map_location, **kwargs)

        _audio_codec_modules.load_fsspec = _load_fsspec_patched
        _audio_codec_modules._hf_load_fsspec_patched = True

    def _resolve_context_audio_path(self, raw_path: str) -> str:
        """Resolve and validate request-provided context path against allowlisted roots."""
        allowed_roots = self.tts_config.context_audio_allowed_roots or []
        if not allowed_roots:
            raise ValueError("context_audio_filepath is disabled; configure context_audio_allowed_roots to enable it.")

        resolved = Path(raw_path).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"context_audio_filepath not found: {resolved}")
        if not resolved.is_file():
            raise ValueError(f"context_audio_filepath is not a file: {resolved}")

        for root in allowed_roots:
            root_resolved = Path(root).expanduser().resolve()
            try:
                resolved.relative_to(root_resolved)
                return str(resolved)
            except ValueError:
                continue

        allowed = ", ".join(str(Path(r).expanduser().resolve()) for r in allowed_roots)
        raise ValueError(f"context_audio_filepath '{resolved}' is outside allowed roots: {allowed}")

    def load_model(self) -> None:
        # Patch NeMo's load_fsspec() to route HuggingFace resolve URLs through
        # huggingface_hub.hf_hub_download() (uses file locks and local caching),
        # avoiding 429s when many ranks start concurrently.
        self._patch_hf_fsspec_loader()

        if (
            _InferenceConfig is None
            or _MagpieInferenceRunner is None
            or _ModelLoadConfig is None
            or _load_magpie_model is None
            or _load_evalset_config is None
        ):
            raise ImportError("Required NeMo MagpieTTS inference modules are not available in this environment.")

        if not self.tts_config.codec_model_path:
            raise ValueError("codec_model_path required")

        # Support both checkpoint mode (hparams + ckpt) and nemo mode
        has_ckpt_mode = self.tts_config.hparams_file and self.tts_config.checkpoint_file
        if has_ckpt_mode:
            cfg = _ModelLoadConfig(
                hparams_file=self.tts_config.hparams_file,
                checkpoint_file=self.tts_config.checkpoint_file,
                codecmodel_path=self.tts_config.codec_model_path,
                legacy_codebooks=self.tts_config.legacy_codebooks,
                legacy_text_conditioning=self.tts_config.legacy_text_conditioning,
                hparams_from_wandb=self.tts_config.hparams_from_wandb,
            )
        else:
            cfg = _ModelLoadConfig(
                nemo_file=self.config.model_path,
                codecmodel_path=self.tts_config.codec_model_path,
                legacy_codebooks=self.tts_config.legacy_codebooks,
                legacy_text_conditioning=self.tts_config.legacy_text_conditioning,
            )
        self._model, self._checkpoint_name = _load_magpie_model(cfg, device=self.config.device)

        # Merge args from MagpieTTSConfig into InferenceConfig. NeMo API differs
        # across builds: some use ModelInferenceParameters, others do not expose it.
        model_inference_candidates = dict(self.tts_config.extra_config)
        if self.tts_config.max_decoder_steps is not None:
            model_inference_candidates["max_decoder_steps"] = self.tts_config.max_decoder_steps
        if self.tts_config.temperature is not None:
            model_inference_candidates["temperature"] = self.tts_config.temperature
        if self.tts_config.top_k is not None:
            model_inference_candidates["top_k"] = self.tts_config.top_k
        if self.tts_config.cfg_scale is not None:
            model_inference_candidates["cfg_scale"] = self.tts_config.cfg_scale

        inference_ctor_params = inspect.signature(_InferenceConfig).parameters
        inference_kwargs = {}
        if "batch_size" in inference_ctor_params:
            inference_kwargs["batch_size"] = 16
        if "use_cfg" in inference_ctor_params:
            inference_kwargs["use_cfg"] = self.tts_config.use_cfg
        if "use_local_transformer" in inference_ctor_params:
            inference_kwargs["use_local_transformer"] = self.tts_config.use_local_transformer
        if "apply_attention_prior" in inference_ctor_params:
            inference_kwargs["apply_attention_prior"] = self.tts_config.apply_attention_prior

        if "model_inference_parameters" in inference_ctor_params:
            if _ModelInferenceParameters is not None and is_dataclass(_ModelInferenceParameters):
                mip_fields = {f.name for f in fields(_ModelInferenceParameters)}
                mip_kwargs = {k: v for k, v in model_inference_candidates.items() if k in mip_fields}
                if hasattr(_ModelInferenceParameters, "from_dict"):
                    inference_kwargs["model_inference_parameters"] = _ModelInferenceParameters.from_dict(mip_kwargs)
                else:
                    inference_kwargs["model_inference_parameters"] = _ModelInferenceParameters(**mip_kwargs)
            else:
                # Older/newer NeMo variants can accept dict-like parameters here.
                inference_kwargs["model_inference_parameters"] = model_inference_candidates
        else:
            # Fallback API: inference params are top-level kwargs on InferenceConfig.
            for key, value in model_inference_candidates.items():
                if key in inference_ctor_params:
                    inference_kwargs[key] = value

        try:
            inference_config = _InferenceConfig(**inference_kwargs)
        except TypeError:
            # Minimal fallback for strict config signatures.
            minimal_kwargs = {
                k: v
                for k, v in inference_kwargs.items()
                if k in {"batch_size", "use_cfg", "use_local_transformer", "apply_attention_prior"}
            }
            inference_config = _InferenceConfig(**minimal_kwargs)

        self._runner = _MagpieInferenceRunner(self._model, inference_config)

        self._temp_dir = tempfile.mkdtemp(prefix="magpie_tts_")
        self.tts_config.output_sample_rate = self._model.sample_rate
        self._is_loaded = True

        # Initialize text normalizer if enabled
        if self.tts_config.enable_normalization:
            if _Normalizer is None:
                raise RuntimeError(
                    "Failed to initialize text normalizer while enable_normalization=true: "
                    "nemo_text_processing is not available."
                )
            try:
                self._normalizer = _Normalizer(
                    lang=self.tts_config.normalizer_lang,
                    input_case=self.tts_config.normalizer_input_case,
                )
                logger.info("Text normalizer initialized (lang=%s)", self.tts_config.normalizer_lang)
            except Exception as e:
                raise RuntimeError("Failed to initialize text normalizer while enable_normalization=true.") from e

        logger.info(
            "Loaded MagpieTTS checkpoint=%s sample_rate=%s cfg=%s",
            self._checkpoint_name,
            self._model.sample_rate,
            self.tts_config.use_cfg,
        )

    def _extract_json(self, text: str) -> dict:
        """Extract JSON object from text, skipping non-JSON parts."""
        if not text:
            return {"text": ""}
        idx = text.find("{")
        if idx >= 0:
            try:
                return json.loads(text[idx:])
            except json.JSONDecodeError:
                pass
        return {"text": text}

    def generate(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        if not self._is_loaded:
            return [GenerationResult(error="Model not loaded", request_id=r.request_id) for r in requests]
        if not requests:
            return []

        start_time = time.time()
        batch_dir = os.path.join(self._temp_dir, f"batch_{int(time.time() * 1000)}")
        output_dir = os.path.join(batch_dir, "output")
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Reset KV caches to avoid cross-request shape mismatches
            if self._model is not None:
                decoder = getattr(self._model, "decoder", None)
                if decoder is not None and hasattr(decoder, "reset_cache"):
                    decoder.reset_cache(use_cache=False)

            # Parse requests, extracting JSON from text
            parsed = [self._extract_json(r.text) for r in requests]

            # Create audio_dir with symlinks to all context audio files
            audio_dir = os.path.join(batch_dir, "audio")
            os.makedirs(audio_dir, exist_ok=True)

            manifest_path = os.path.join(batch_dir, "manifest.json")
            with open(manifest_path, "w") as f:
                for i, p in enumerate(parsed):
                    ctx = p.get("context_audio_filepath", "")
                    if ctx:
                        resolved_ctx = self._resolve_context_audio_path(str(ctx))
                        link_name = f"ctx_{i}_{os.path.basename(resolved_ctx)}"
                        link_path = os.path.join(audio_dir, link_name)
                        if not os.path.exists(link_path):
                            os.symlink(resolved_ctx, link_path)
                    else:
                        link_name = f"d{i}.wav"
                        link_path = os.path.join(audio_dir, link_name)
                        if not os.path.exists(link_path):
                            sr = int(getattr(self.tts_config, "output_sample_rate", 22050) or 22050)
                            dur_s = 0.1
                            n = max(1, int(sr * dur_s))
                            sf.write(link_path, [0.0] * n, sr)
                    text = p.get("text", "")
                    if self._normalizer:
                        try:
                            text = self._normalizer.normalize(text, punct_pre_process=True, punct_post_process=True)
                        except Exception as e:
                            raise RuntimeError(f"Failed to normalize text for sample index {i}") from e
                    entry = {
                        "text": text,
                        "audio_filepath": link_name,
                        "context_audio_filepath": link_name,
                        "duration": p.get("duration", 5.0),
                        "context_audio_duration": p.get("context_audio_duration", 5.0),
                    }
                    if p.get("speaker_index") is not None:
                        entry["speaker_index"] = int(p["speaker_index"])
                    f.write(json.dumps(entry) + "\n")

            config_path = os.path.join(batch_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump({"batch": {"manifest_path": manifest_path, "audio_dir": audio_dir}}, f)

            # Run inference
            dataset = self._runner.create_dataset(_load_evalset_config(config_path))
            rtf_list, _, _ = self._runner.run_inference_on_dataset(
                dataset, output_dir, save_cross_attention_maps=False, save_context_audio=False
            )

            gen_time = time.time() - start_time
            batch_metrics = {
                "total_time_sec": gen_time,
                "num_samples": len(requests),
                **self._runner.compute_mean_rtf_metrics(rtf_list),
            }

            # Build results
            results = []
            for i, req in enumerate(requests):
                path = os.path.join(output_dir, f"predicted_audio_{i}.wav")
                if os.path.exists(path):
                    audio, sr = sf.read(path)
                    buf = io.BytesIO()
                    sf.write(buf, audio, sr, format="WAV")
                    buf.seek(0)
                    dur = len(audio) / sr
                    results.append(
                        GenerationResult(
                            text=parsed[i].get("text", ""),
                            audio_bytes=buf.read(),
                            audio_sample_rate=self.tts_config.output_sample_rate,
                            audio_format="wav",
                            request_id=req.request_id,
                            generation_time_ms=gen_time * 1000 / len(requests),
                            debug_info={
                                "checkpoint": self._checkpoint_name,
                                "audio_duration_sec": dur,
                                "rtf": gen_time / len(requests) / dur if dur else 0,
                                "config": {
                                    "temp": self.tts_config.temperature,
                                    "top_k": self.tts_config.top_k,
                                    "cfg": self.tts_config.use_cfg,
                                    "cfg_scale": self.tts_config.cfg_scale,
                                },
                                "batch_metrics": batch_metrics,
                            },
                        )
                    )
                else:
                    results.append(GenerationResult(error=f"Audio not found: {path}", request_id=req.request_id))
            return results
        except Exception as e:
            logger.exception("Magpie generation failed")
            return [GenerationResult(error=str(e), request_id=r.request_id) for r in requests]
        finally:
            shutil.rmtree(batch_dir, ignore_errors=True)

    def validate_request(self, request: GenerationRequest) -> Optional[str]:
        return "Text required" if not request.text else None

    def health_check(self) -> Dict[str, Any]:
        h = super().health_check()
        if self._is_loaded:
            h.update(
                {
                    "checkpoint": self._checkpoint_name,
                    "codec": self.tts_config.codec_model_path,
                    "cfg": self.tts_config.use_cfg,
                    "cfg_scale": self.tts_config.cfg_scale,
                    "sample_rate": self.tts_config.output_sample_rate,
                }
            )
        return h

    def __del__(self):
        if getattr(self, "_temp_dir", None) and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
