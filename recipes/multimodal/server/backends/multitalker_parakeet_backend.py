# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Licensed under the Apache License, Version 2.0

"""Unified-server backend for persistent Multitalker Parakeet streaming ASR.

This backend exposes the official NeMo multitalker streaming path through the
OpenAI-compatible ``serve_unified`` server.  Model weights are loaded once in
``load_model()`` and each request batch is run in-process against those model
instances.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
import tempfile
import threading
import time
import wave
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .base import BackendConfig, GenerationRequest, GenerationResult, InferenceBackend, Modality

logger = logging.getLogger(__name__)

DEFAULT_ASR_MODEL = "nvidia/multitalker-parakeet-streaming-0.6b-v1"
DEFAULT_DIAR_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"
DEFAULT_NEMO_ROOT = "/opt/NeMo"


@dataclass
class _RuntimeModules:
    torch: Any
    pl: Any
    nemo_asr: Any
    SortformerEncLabelModel: Any
    CacheAwareStreamingAudioBuffer: Any
    get_multi_talker_samples_from_manifest: Any
    OmegaConf: Any


@dataclass
class MultitalkerParakeetConfig(BackendConfig):
    """Configuration for the Multitalker Parakeet unified backend."""

    model_name: Optional[str] = None
    asr_model: Optional[str] = None
    diar_model: str = DEFAULT_DIAR_MODEL
    nemo_root: str = DEFAULT_NEMO_ROOT
    script_path: str = ""
    model_cache_dir: Optional[str] = None
    resolve_hf_models: bool = True
    spk_supervision: str = "diar"
    max_num_of_spks: int = 4
    masked_asr: bool = False
    parallel_speaker_strategy: bool = True
    binary_diar_preds: bool = True
    batch_size: int = 16
    mt_batch_size: Optional[int] = None
    num_workers: int = 0
    att_context_size: Any = "[70,13]"
    spkcache_len: int = 188
    spkcache_refresh_rate: int = 144
    fifo_len: int = 188
    chunk_len: int = 13
    chunk_left_context: int = 1
    chunk_right_context: int = 0
    cache_gating: bool = False
    request_timeout_s: int = 7200
    log: bool = False
    session_len_sec: float = -1.0
    streaming_mode: bool = True
    use_amp: bool = True
    online_normalization: bool = False
    pad_and_drop_preencoded: bool = False
    chunk_size: int = -1
    shift_size: int = -1
    left_chunks: int = 2
    matmul_precision: str = "highest"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MultitalkerParakeetConfig":
        if d.get("model_name") and not d.get("model_path"):
            d = {**d, "model_path": d["model_name"]}
        if d.get("asr_model") and not d.get("model_path"):
            d = {**d, "model_path": d["asr_model"]}
        if d.get("model_path") and not d.get("asr_model"):
            d = {**d, "asr_model": d["model_path"]}
        if d.get("mt_batch_size"):
            d = {**d, "batch_size": d["mt_batch_size"]}

        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(
            **{k: v for k, v in d.items() if k in known and k != "extra_config"},
            extra_config={k: v for k, v in d.items() if k not in known},
        )


def _resolve_script_path(cfg: MultitalkerParakeetConfig) -> Path:
    if cfg.script_path:
        return Path(cfg.script_path)
    return Path(cfg.nemo_root) / "examples/asr/asr_cache_aware_streaming/speech_to_text_multitalker_streaming_infer.py"


def _session_id_from_audio_path(path: str) -> str:
    return Path(path).stem


def _duration_seconds(path: Path) -> float:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        return float(info.frames) / float(info.samplerate)
    except Exception:
        with wave.open(str(path), "rb") as wavf:
            return float(wavf.getnframes()) / float(wavf.getframerate())


def _get_request_audio_bytes(request: GenerationRequest) -> bytes:
    if request.audio_bytes:
        return request.audio_bytes
    if request.audio_bytes_list:
        if len(request.audio_bytes_list) > 1:
            raise ValueError("multitalker_parakeet backend supports one mixed audio input per request.")
        return request.audio_bytes_list[0]
    raise ValueError("Request must contain audio_bytes or audio_bytes_list.")


def _seglst_to_sot(entries: list[dict]) -> str:
    tag_by_speaker: dict[str, str] = {}
    chunks: list[str] = []
    for entry in sorted(
        entries,
        key=lambda item: (
            float(item.get("start_time", 0.0) or 0.0),
            float(item.get("end_time", 0.0) or 0.0),
            str(item.get("speaker", "")),
        ),
    ):
        words = str(entry.get("words") or entry.get("text") or "").strip()
        if not words:
            continue
        speaker = str(entry.get("speaker") or "speaker")
        if speaker not in tag_by_speaker:
            tag_by_speaker[speaker] = f"s{len(tag_by_speaker)}"
        chunks.append(f"[{tag_by_speaker[speaker]}] {words}")
    return " ".join(chunks)


def _group_seglst_by_session(entries: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        session_id = str(entry.get("session_id") or entry.get("recording_id") or entry.get("audio_id") or "")
        if not session_id and entry.get("audio_filepath"):
            session_id = _session_id_from_audio_path(str(entry["audio_filepath"]))
        grouped[session_id].append(entry)
    return grouped


def _choose_nemo_file(repo_id: str, files: list[str], preferred_name: str | None) -> str | None:
    nemo_files = [name for name in files if name.endswith(".nemo")]
    if not nemo_files:
        return None
    if preferred_name:
        preferred = [name for name in nemo_files if Path(name).name == preferred_name]
        if preferred:
            return preferred[0]
    repo_basename = repo_id.rsplit("/", 1)[-1]
    for name in nemo_files:
        if Path(name).stem == repo_basename:
            return name
    return nemo_files[0]


def _resolve_hf_nemo_file(model_ref: str, *, cache_dir: str | None = None, required: bool = False) -> str:
    model_path = Path(model_ref)
    if model_path.exists():
        return str(model_path)
    if model_ref.endswith((".nemo", ".ckpt")) and os.sep in model_ref:
        return model_ref
    if "/" not in model_ref:
        return model_ref

    try:
        from huggingface_hub import hf_hub_download, list_repo_files

        files = list_repo_files(model_ref, repo_type="model")
        chosen = _choose_nemo_file(model_ref, files, preferred_name=f"{model_ref.rsplit('/', 1)[-1]}.nemo")
        if chosen is None:
            if required:
                raise FileNotFoundError(f"No .nemo file found in Hugging Face repo {model_ref}")
            return model_ref
        return hf_hub_download(repo_id=model_ref, filename=chosen, cache_dir=cache_dir)
    except Exception as exc:
        if required:
            raise RuntimeError(f"Could not resolve required Hugging Face .nemo file for {model_ref}: {exc}") from exc
        logger.warning("Could not resolve Hugging Face model %s to a .nemo file: %s", model_ref, exc)
        return model_ref


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _as_int(value: Any) -> int:
    return int(value)


def _parse_att_context_size(value: Any) -> list[int] | None:
    if value in (None, "", "None"):
        return None
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            return [int(item) for item in json.loads(text)]
        return [int(item.strip()) for item in text.split(",") if item.strip()]
    raise TypeError(f"Unsupported att_context_size value: {value!r}")


def _import_script_module(script_path: Path) -> Any:
    module_name = "_nemo_multitalker_streaming_infer"
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import Multitalker Parakeet script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _import_runtime_modules() -> _RuntimeModules:
    import pytorch_lightning as pl
    import torch
    from omegaconf import OmegaConf

    import nemo.collections.asr as nemo_asr
    from nemo.collections.asr.models.sortformer_diar_models import SortformerEncLabelModel
    from nemo.collections.asr.parts.utils.multispk_transcribe_utils import get_multi_talker_samples_from_manifest
    from nemo.collections.asr.parts.utils.streaming_utils import CacheAwareStreamingAudioBuffer

    return _RuntimeModules(
        torch=torch,
        pl=pl,
        nemo_asr=nemo_asr,
        SortformerEncLabelModel=SortformerEncLabelModel,
        CacheAwareStreamingAudioBuffer=CacheAwareStreamingAudioBuffer,
        get_multi_talker_samples_from_manifest=get_multi_talker_samples_from_manifest,
        OmegaConf=OmegaConf,
    )


def _configure_cpu_thread_env() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(name, "1")


class MultitalkerParakeetBackend(InferenceBackend):
    """Unified-server backend for persistent NeMo Multitalker Parakeet inference."""

    @classmethod
    def get_config_class(cls) -> type:
        return MultitalkerParakeetConfig

    @property
    def name(self) -> str:
        return "multitalker_parakeet"

    @property
    def supported_modalities(self) -> Set[Modality]:
        return {Modality.AUDIO_IN, Modality.TEXT}

    def __init__(self, config: BackendConfig):
        self.mt_config = (
            config if isinstance(config, MultitalkerParakeetConfig) else MultitalkerParakeetConfig.from_dict(config.extra_config)
        )
        super().__init__(self.mt_config)
        self._script_path: Path | None = None
        self._script_module: Any = None
        self._runtime: _RuntimeModules | None = None
        self._runtime_cfg_template: Any = None
        self._asr_model_ref = self.mt_config.asr_model or self.mt_config.model_path or DEFAULT_ASR_MODEL
        self._diar_model_ref = self.mt_config.diar_model
        self._asr_model = None
        self._diar_model = None
        self._device = self.mt_config.device
        self._pl_devices: Any = 1
        self._accelerator = "cpu"
        self._inference_lock = threading.Lock()

    def load_model(self) -> None:
        _configure_cpu_thread_env()
        script_path = _resolve_script_path(self.mt_config)
        if not script_path.exists():
            raise FileNotFoundError(f"Multitalker Parakeet script not found: {script_path}")
        self._script_path = script_path
        self._script_module = _import_script_module(script_path)
        self._runtime = _import_runtime_modules()
        try:
            self._runtime.torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "1"))))
        except Exception as exc:
            logger.warning("Could not set torch CPU thread count: %s", exc)

        if self.mt_config.resolve_hf_models:
            self._asr_model_ref = _resolve_hf_nemo_file(
                self._asr_model_ref,
                cache_dir=self.mt_config.model_cache_dir,
            )
            self._diar_model_ref = _resolve_hf_nemo_file(
                self._diar_model_ref,
                cache_dir=self.mt_config.model_cache_dir,
                required=not str(self._diar_model_ref).endswith((".nemo", ".ckpt")),
            )

        self._runtime_cfg_template = self._build_runtime_cfg()
        self._device, self._pl_devices, self._accelerator, map_location = self._select_device()
        self._runtime.torch.set_float32_matmul_precision(str(self._runtime_cfg_template.matmul_precision))

        self._diar_model = self._load_diar_model(map_location)
        trainer = self._runtime.pl.Trainer(devices=self._pl_devices, accelerator=self._accelerator)
        self._diar_model.set_trainer(trainer)
        self._configure_diar_model(self._runtime_cfg_template)
        self._diar_model = self._diar_model.eval()

        self._asr_model = self._load_asr_model(map_location)
        self._asr_model = self._asr_model.to(self._device)
        self._asr_model.eval()
        self._configure_asr_model(self._runtime_cfg_template)
        self._set_script_autocast(self._runtime_cfg_template)

        self._model = self._asr_model
        self._is_loaded = True
        logger.info(
            "Loaded Multitalker Parakeet backend with asr_model=%s diar_model=%s script=%s device=%s",
            self._asr_model_ref,
            self._diar_model_ref,
            self._script_path,
            self._device,
        )

    def _build_runtime_cfg(self) -> Any:
        cfg = self._runtime.OmegaConf.structured(self._script_module.MultitalkerTranscriptionConfig())
        cfg.asr_model = str(self._asr_model_ref)
        cfg.diar_model = str(self._diar_model_ref)
        cfg.diar_pretrained_name = None
        cfg.device = str(self.mt_config.device)
        cfg.manifest_file = None
        cfg.audio_file = None
        cfg.log = _as_bool(self.mt_config.log)
        cfg.binary_diar_preds = _as_bool(self.mt_config.binary_diar_preds)
        cfg.spk_supervision = str(self.mt_config.spk_supervision)
        cfg.max_num_of_spks = _as_int(self.mt_config.max_num_of_spks)
        cfg.masked_asr = _as_bool(self.mt_config.masked_asr)
        cfg.parallel_speaker_strategy = _as_bool(self.mt_config.parallel_speaker_strategy)
        cfg.att_context_size = _parse_att_context_size(self.mt_config.att_context_size)
        cfg.generate_realtime_scripts = False
        cfg.batch_size = max(1, _as_int(self.mt_config.batch_size))
        cfg.num_workers = max(0, _as_int(self.mt_config.num_workers))
        cfg.cache_gating = _as_bool(self.mt_config.cache_gating)
        cfg.spkcache_len = _as_int(self.mt_config.spkcache_len)
        cfg.spkcache_refresh_rate = _as_int(self.mt_config.spkcache_refresh_rate)
        cfg.fifo_len = _as_int(self.mt_config.fifo_len)
        cfg.chunk_len = _as_int(self.mt_config.chunk_len)
        cfg.chunk_left_context = _as_int(self.mt_config.chunk_left_context)
        cfg.chunk_right_context = _as_int(self.mt_config.chunk_right_context)
        cfg.session_len_sec = float(self.mt_config.session_len_sec)
        cfg.streaming_mode = _as_bool(self.mt_config.streaming_mode)
        cfg.use_amp = _as_bool(self.mt_config.use_amp)
        cfg.online_normalization = _as_bool(self.mt_config.online_normalization)
        cfg.pad_and_drop_preencoded = _as_bool(self.mt_config.pad_and_drop_preencoded)
        cfg.chunk_size = _as_int(self.mt_config.chunk_size)
        cfg.shift_size = _as_int(self.mt_config.shift_size)
        cfg.left_chunks = _as_int(self.mt_config.left_chunks)
        cfg.matmul_precision = str(self.mt_config.matmul_precision)
        return cfg

    def _select_device(self) -> tuple[str, Any, str, Any]:
        requested = str(self.mt_config.device or "cuda")
        torch = self._runtime.torch
        if requested.startswith("cuda") and torch.cuda.is_available():
            if ":" in requested:
                device_idx = int(requested.split(":", 1)[1])
            else:
                device_idx = 0
            device = f"cuda:{device_idx}"
            return device, [device_idx], "gpu", torch.device(device)
        return "cpu", 1, "cpu", torch.device("cpu")

    def _load_diar_model(self, map_location: Any) -> Any:
        model_ref = str(self._diar_model_ref)
        if model_ref.endswith(".ckpt"):
            return self._runtime.SortformerEncLabelModel.load_from_checkpoint(
                checkpoint_path=model_ref,
                map_location=map_location,
                strict=False,
            )
        if model_ref.endswith(".nemo") or Path(model_ref).exists():
            return self._runtime.SortformerEncLabelModel.restore_from(restore_path=model_ref, map_location=map_location)
        return self._runtime.SortformerEncLabelModel.from_pretrained(model_ref)

    def _load_asr_model(self, map_location: Any) -> Any:
        model_ref = str(self._asr_model_ref)
        if model_ref.endswith(".nemo") or Path(model_ref).exists():
            try:
                return self._runtime.nemo_asr.models.ASRModel.restore_from(
                    restore_path=model_ref,
                    map_location=map_location,
                )
            except TypeError:
                return self._runtime.nemo_asr.models.ASRModel.restore_from(restore_path=model_ref)
        try:
            return self._runtime.nemo_asr.models.ASRModel.from_pretrained(
                model_name=model_ref,
                map_location=map_location,
            )
        except TypeError:
            return self._runtime.nemo_asr.models.ASRModel.from_pretrained(model_name=model_ref)

    def _configure_diar_model(self, cfg: Any) -> None:
        test_ds = self._diar_model._cfg.test_ds
        test_ds.session_len_sec = cfg.session_len_sec
        test_ds.batch_size = cfg.batch_size
        test_ds.num_workers = cfg.num_workers

        self._diar_model.streaming_mode = cfg.streaming_mode
        modules = getattr(self._diar_model, "sortformer_modules", None)
        if modules is not None:
            modules.chunk_len = cfg.chunk_len
            modules.spkcache_len = cfg.spkcache_len
            modules.chunk_left_context = cfg.chunk_left_context
            modules.chunk_right_context = cfg.chunk_right_context
            modules.fifo_len = cfg.fifo_len
            modules.log = cfg.log
            modules.spkcache_refresh_rate = cfg.spkcache_refresh_rate

    def _configure_asr_model(self, cfg: Any) -> None:
        if cfg.att_context_size is not None:
            if hasattr(self._asr_model.encoder, "set_default_att_context_size"):
                self._asr_model.encoder.set_default_att_context_size(att_context_size=cfg.att_context_size)
            else:
                raise ValueError("Model does not support multiple lookaheads.")

        if cfg.chunk_size > 0:
            shift_size = cfg.chunk_size if cfg.shift_size < 0 else cfg.shift_size
            self._asr_model.encoder.setup_streaming_params(
                chunk_size=cfg.chunk_size,
                left_chunks=cfg.left_chunks,
                shift_size=shift_size,
            )

    def _set_script_autocast(self, cfg: Any) -> None:
        device = getattr(self._asr_model, "device", None)
        device_type = getattr(device, "type", None) or str(self._device).split(":", 1)[0]
        self._script_module.autocast = self._runtime.torch.amp.autocast(device_type, enabled=cfg.use_amp)

    def validate_request(self, request: GenerationRequest) -> Optional[str]:
        has_audio = request.audio_bytes is not None or (
            request.audio_bytes_list is not None and len(request.audio_bytes_list) > 0
        )
        if not has_audio:
            return "multitalker_parakeet backend requires audio input"
        if request.audio_bytes_list is not None and len(request.audio_bytes_list) > 1:
            return "multitalker_parakeet backend supports one mixed audio input per request"
        return None

    def _batch_cfg(self, manifest_file: Path, batch_size: int) -> Any:
        payload = self._runtime.OmegaConf.to_container(self._runtime_cfg_template, resolve=True)
        cfg = self._runtime.OmegaConf.create(payload)
        cfg.manifest_file = str(manifest_file)
        cfg.audio_file = None
        cfg.batch_size = max(1, min(int(cfg.batch_size), max(1, batch_size)))
        cfg.device = self._device
        return cfg

    def _setup_diar_test_data(self, cfg: Any) -> None:
        test_ds = self._diar_model._cfg.test_ds
        test_ds.session_len_sec = cfg.session_len_sec
        test_ds.manifest_filepath = cfg.manifest_file
        test_ds.batch_size = cfg.batch_size
        test_ds.num_workers = cfg.num_workers
        self._diar_model.setup_test_data(test_data_config=test_ds)

    @staticmethod
    def _json_scalar(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if hasattr(value, "item"):
            return value.item()
        return str(value)

    @classmethod
    def _sanitize_seglst_entry(cls, entry: dict) -> dict:
        sanitized: dict[str, Any] = {}
        for key in ("session_id", "recording_id", "audio_id", "audio_filepath", "speaker", "words", "text"):
            if key in entry:
                sanitized[key] = str(cls._json_scalar(entry[key]))
        for key in ("start_time", "end_time"):
            if key in entry:
                sanitized[key] = float(cls._json_scalar(entry[key]))
        return sanitized

    @staticmethod
    def _coerce_batch_entries(streamer: Any, batch_entries: Any) -> list[dict]:
        if batch_entries is not None:
            entries = list(batch_entries)
        else:
            instance_manager = getattr(streamer, "instance_manager", None)
            entries = list(getattr(instance_manager, "seglst_dict_list", None) or [])
        return [
            MultitalkerParakeetBackend._sanitize_seglst_entry(entry)
            for entry in entries
            if isinstance(entry, dict)
        ]

    def _run_inprocess_manifest(self, manifest_file: Path, batch_size: int) -> list[dict]:
        if self._runtime is None or self._script_module is None:
            raise RuntimeError("Multitalker backend runtime is not initialized.")

        cfg = self._batch_cfg(manifest_file, batch_size)
        with self._inference_lock:
            self._setup_diar_test_data(cfg)
            self._set_script_autocast(cfg)

            feat_per_sec = round(
                self._asr_model.cfg.preprocessor.window_stride * self._asr_model.cfg.encoder.subsampling_factor,
                2,
            )
            samples, rttms_mask_mats = self._runtime.get_multi_talker_samples_from_manifest(
                cfg,
                manifest_file=str(manifest_file),
                feat_per_sec=feat_per_sec,
                max_spks=cfg.max_num_of_spks,
            )
            if cfg.spk_supervision == "rttm":
                self._diar_model.add_rttms_mask_mats(rttms_mask_mats, device=self._asr_model.device)

            logger.info("Running persistent Multitalker Parakeet inference for %d samples", len(samples))
            streaming_buffer = self._runtime.CacheAwareStreamingAudioBuffer(
                model=self._asr_model,
                online_normalization=cfg.online_normalization,
                pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
            )

            seglst_entries: list[dict] = []
            batch_samples: list[dict] = []
            for sample_idx, sample in enumerate(samples):
                batch_samples.append(sample)
                streaming_buffer.append_audio_file(sample["audio_filepath"], stream_id=-1)

                if (sample_idx + 1) % cfg.batch_size == 0 or sample_idx == len(samples) - 1:
                    if cfg.parallel_speaker_strategy:
                        streamer = self._script_module.launch_parallel_streaming(
                            cfg=cfg,
                            asr_model=self._asr_model,
                            diar_model=self._diar_model,
                            streaming_buffer=streaming_buffer,
                            pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
                        )
                        batch_entries = streamer.generate_seglst_dicts_from_parallel_streaming(samples=batch_samples)
                    else:
                        streamer = self._script_module.launch_serial_streaming(
                            cfg=cfg,
                            asr_model=self._asr_model,
                            diar_model=self._diar_model,
                            streaming_buffer=streaming_buffer,
                            pad_and_drop_preencoded=cfg.pad_and_drop_preencoded,
                        )
                        batch_entries = streamer.generate_seglst_dicts_from_serial_streaming(samples=batch_samples)

                    seglst_entries.extend(self._coerce_batch_entries(streamer, batch_entries))
                    streaming_buffer.reset_buffer()
                    batch_samples = []

            return seglst_entries

    def _manifest_record(self, audio_path: Path, request: GenerationRequest) -> dict[str, Any]:
        extra = request.extra_params or {}
        record = {
            "audio_filepath": str(audio_path),
            "offset": float(extra.get("offset", 0.0) or 0.0),
            "duration": float(extra.get("duration") or round(_duration_seconds(audio_path), 4)),
        }
        if extra.get("rttm_filepath"):
            record["rttm_filepath"] = str(extra["rttm_filepath"])
        return record

    def generate(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        if not self._is_loaded:
            return [GenerationResult(error="Model not loaded", request_id=request.request_id) for request in requests]
        if not requests:
            return []

        start = time.time()
        session_ids: list[str] = []
        results: list[GenerationResult | None] = [None] * len(requests)

        with tempfile.TemporaryDirectory(prefix="multitalker_parakeet_") as temp_name:
            temp_dir = Path(temp_name)
            manifest_file = temp_dir / "input.jsonl"

            with open(manifest_file, "wt", encoding="utf-8") as fout:
                for idx, request in enumerate(requests):
                    try:
                        audio_path = temp_dir / f"req_{idx:04d}.wav"
                        audio_path.write_bytes(_get_request_audio_bytes(request))
                        session_ids.append(audio_path.stem)
                        fout.write(json.dumps(self._manifest_record(audio_path, request)) + "\n")
                    except Exception as exc:
                        session_ids.append("")
                        results[idx] = GenerationResult(error=str(exc), request_id=request.request_id)

            runnable_indices = [idx for idx, result in enumerate(results) if result is None]
            if not runnable_indices:
                return [result if result is not None else GenerationResult(error="Unknown multitalker backend error") for result in results]

            try:
                entries = self._run_inprocess_manifest(manifest_file, len(runnable_indices))
                grouped_entries = _group_seglst_by_session(entries)
                per_req_ms = (time.time() - start) * 1000.0 / max(len(runnable_indices), 1)

                for idx in runnable_indices:
                    session_entries = grouped_entries.get(session_ids[idx], [])
                    text = _seglst_to_sot(session_entries)
                    results[idx] = GenerationResult(
                        text=text,
                        request_id=requests[idx].request_id,
                        generation_time_ms=per_req_ms,
                        debug_info={
                            "backend": self.name,
                            "model": self._asr_model_ref,
                            "diar_model": self._diar_model_ref,
                            "spk_supervision": self.mt_config.spk_supervision,
                            "hypothesis_seglst": session_entries,
                            "persistent_inprocess": True,
                        },
                    )
            except Exception as exc:
                logger.exception("Persistent Multitalker Parakeet inference failed")
                error = str(exc)
                for idx in runnable_indices:
                    results[idx] = GenerationResult(error=error, request_id=requests[idx].request_id)

            return [result if result is not None else GenerationResult(error="Unknown multitalker backend error") for result in results]
