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

"""
V2 Incremental Speech-to-Speech (S2S) backend.

Wraps NemotronVoicechatInferenceWrapper from the NeMo inference pipeline
instead of manually implementing model loading and inference. This enables:
  - vLLM acceleration for both LLM and EarTTS
  - Perception cache (incremental encoder)
  - Codec cache (incremental audio decoding, no clicking)
  - System prompt prefill
  - Proper token-to-text BPE decoding

All model initialization, weight loading, and core inference logic is
delegated to the wrapper -- this backend only adapts the InferenceBackend
server interface around it.
"""

import io
import json
import os
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .base import (
    BackendConfig,
    GenerationRequest,
    GenerationResult,
    InferenceBackend,
    Modality,
)

SAMPLE_RATE = 16000
FRAME_SIZE_SEC = 0.08
FRAME_SIZE_SAMPLES = int(SAMPLE_RATE * FRAME_SIZE_SEC)
TTS_SAMPLE_RATE = 22050

DEFAULT_BUFFER_SIZE_FRAMES = 71
DEFAULT_NUM_FRAMES_PER_INFERENCE = 1
DEFAULT_CODEC_TOKEN_HISTORY_SIZE = 600


@dataclass
class S2SIncrementalV2Config(BackendConfig):
    """Configuration for V2 incremental S2S backend.

    Extends the base BackendConfig with parameters for the NeMo inference
    pipeline wrapper, including vLLM, perception/codec caches, and prompts.
    """

    config_path: Optional[str] = None

    llm_checkpoint_path: Optional[str] = None
    tts_checkpoint_path: Optional[str] = None
    speaker_reference: Optional[str] = None

    buffer_size_frames: int = DEFAULT_BUFFER_SIZE_FRAMES
    num_frames_per_inference: int = DEFAULT_NUM_FRAMES_PER_INFERENCE
    codec_token_history_size: int = DEFAULT_CODEC_TOKEN_HISTORY_SIZE

    # Padding: pad_to_duration_secs takes precedence; if unset, silence_padding_sec
    # seconds of silence are appended (matching V1 behavior).
    silence_padding_sec: float = 5.0
    pad_to_duration_secs: Optional[float] = None

    force_turn_taking: bool = False
    force_turn_taking_threshold: int = 40
    force_turn_taking_pad_window: int = 25

    decode_audio: bool = True
    merge_user_channel: bool = False
    use_asr_as_response: bool = False

    save_session_artifacts: bool = True
    session_artifacts_dir: str = "/tmp/s2s_sessions"

    output_frame_alignment: bool = False

    response_end_detection_mode: str = "audio_energy"
    audio_energy_threshold: float = 0.01
    audio_energy_window_sec: float = 0.5
    max_response_duration_sec: float = 30.0
    eos_detection_window: int = 10

    engine_type: str = "native"

    use_perception_cache: bool = False
    use_perception_cudagraph: bool = False

    use_codec_cache: bool = True

    repetition_penalty: float = 1.0
    top_p: float = 1.0
    temperature: float = 1.0

    inference_pad_boost: Optional[float] = None
    inference_bos_boost: Optional[float] = None
    inference_eos_boost: Optional[float] = None
    inference_user_pad_boost: Optional[float] = None
    inference_user_bos_boost: Optional[float] = None
    inference_user_eos_boost: Optional[float] = None

    system_prompt: Optional[str] = None
    tts_system_prompt: Optional[str] = None

    vllm_llm_config: Optional[Dict[str, Any]] = field(default_factory=lambda: None)
    vllm_tts_config: Optional[Dict[str, Any]] = field(default_factory=lambda: None)

    matmul_precision: str = "high"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "S2SIncrementalV2Config":
        known_fields = {f.name for f in cls.__dataclass_fields__.values() if f.name != "extra_config"}
        known = {k: v for k, v in d.items() if k in known_fields}
        extra = {k: v for k, v in d.items() if k not in known_fields}
        return cls(**known, extra_config=extra)


class S2SIncrementalBackendV2(InferenceBackend):
    """
    V2 Incremental Speech-to-Speech backend.

    Wraps ``NemotronVoicechatInferenceWrapper`` for model loading,
    frame-level inference, and audio decoding. Adds the
    ``InferenceBackend`` interface required by the unified server.
    """

    @property
    def name(self) -> str:
        return "s2s_incremental_v2"

    @property
    def supported_modalities(self) -> Set[Modality]:
        return {Modality.TEXT, Modality.AUDIO_IN, Modality.AUDIO_OUT}

    def __init__(self, config: BackendConfig):
        if isinstance(config, S2SIncrementalV2Config):
            self.v2_config = config
        else:
            self.v2_config = S2SIncrementalV2Config.from_dict(
                {
                    **{
                        k: getattr(config, k)
                        for k in [
                            "model_path",
                            "device",
                            "dtype",
                            "max_new_tokens",
                            "temperature",
                            "top_p",
                            "top_k",
                        ]
                    },
                    **config.extra_config,
                }
            )

        # Alias so code written against V1's inc_config keeps working
        self.inc_config = self.v2_config

        super().__init__(self.v2_config)

        self._wrapper = None
        self._tokenizer = None
        self._model_cfg = None
        self.dtype = None

        self.first_context_subword_id = None
        self.generation_config = None
        self.first_tts_code_input = None
        self.first_tts_past_key_values_input = None
        self.target_sample_rate = TTS_SAMPLE_RATE
        self.target_fps = None

    # ------------------------------------------------------------------
    # Wrapper config builder
    # ------------------------------------------------------------------
    def _build_wrapper_config(self) -> DictConfig:
        """Translate ``S2SIncrementalV2Config`` into the ``DictConfig``
        expected by ``NemotronVoicechatInferenceWrapper``."""
        cfg = self.v2_config
        model_path = cfg.tts_checkpoint_path or cfg.model_path
        llm_path = cfg.llm_checkpoint_path or cfg.model_path

        d: Dict[str, Any] = {
            "model_path": model_path,
            "llm_checkpoint_path": llm_path,
            "speaker_reference": cfg.speaker_reference,
            "buffer_size_frames": cfg.buffer_size_frames,
            "decode_audio": cfg.decode_audio,
            "codec_token_history_size": cfg.codec_token_history_size,
            "engine_type": cfg.engine_type,
            "use_perception_cache": cfg.use_perception_cache,
            "use_perception_cudagraph": cfg.use_perception_cudagraph,
            "use_codec_cache": cfg.use_codec_cache,
            "top_p": cfg.top_p,
            "repetition_penalty": cfg.repetition_penalty,
            "temperature": cfg.temperature,
            "tts_system_prompt": cfg.tts_system_prompt,
            "compute_dtype": cfg.dtype,
            "device": cfg.device,
            "force_turn_taking": cfg.force_turn_taking,
            "force_turn_taking_threshold": cfg.force_turn_taking_threshold,
            "force_turn_taking_pad_window": cfg.force_turn_taking_pad_window,
        }

        for boost_key in (
            "inference_pad_boost",
            "inference_bos_boost",
            "inference_eos_boost",
            "inference_user_pad_boost",
            "inference_user_bos_boost",
            "inference_user_eos_boost",
        ):
            val = getattr(cfg, boost_key, None)
            if val is not None:
                d[boost_key] = val

        if cfg.vllm_llm_config:
            d["vllm_llm_config"] = cfg.vllm_llm_config
        if cfg.vllm_tts_config:
            d["vllm_tts_config"] = cfg.vllm_tts_config

        return OmegaConf.create(d)

    # ------------------------------------------------------------------
    # Model loading -- delegates entirely to the wrapper
    # ------------------------------------------------------------------
    def load_model(self) -> None:
        from nemo.collections.speechlm2.inference.model_wrappers.nemotron_voicechat_inference_wrapper import (
            NemotronVoicechatInferenceWrapper,
        )

        print(f"[S2SIncrementalV2] Loading model (engine={self.v2_config.engine_type})...")

        torch.set_float32_matmul_precision(self.v2_config.matmul_precision)

        model_cfg = self._build_wrapper_config()
        self._wrapper = NemotronVoicechatInferenceWrapper(model_cfg=model_cfg)

        # Expose attributes expected by the server and session backend
        self._model = self._wrapper.model
        self._tokenizer = self._wrapper.tokenizer
        self._model_cfg = self._wrapper.model_cfg
        self.dtype = self._wrapper.dtype
        self.target_sample_rate = getattr(self._wrapper, "target_sample_rate", TTS_SAMPLE_RATE)
        self.target_fps = getattr(self._wrapper, "target_fps", None)

        self.first_context_subword_id = self._wrapper.first_context_subword_id
        self.generation_config = self._wrapper.generation_config
        self.first_tts_code_input = self._wrapper.first_tts_code_input
        self.first_tts_past_key_values_input = self._wrapper.first_tts_past_key_values_input

        self._is_loaded = True
        print("[S2SIncrementalV2] Model loaded successfully")

    # ------------------------------------------------------------------
    # Core inference -- thin delegates to wrapper
    # ------------------------------------------------------------------
    def infer_one_step(
        self,
        audio_input,
        num_frames_per_chunk=None,
        num_frames_per_inference=None,
        frame_idx=0,
        gen_text=None,
        audio_toks_buffer=None,
        input_embeds_history=None,
        dynamic_cache=None,
        embedding_position=None,  # accepted for V1 compat, not forwarded
        past_key_values=None,
        code=None,
        subword_mask=None,
        gen_asr_text=None,
        request_id=None,
        perception_cache=None,
        has_prompt=False,
        codec_cache=None,
    ):
        """Delegate to the wrapper's ``infer_one_step``.

        Accepts both ``num_frames_per_chunk`` (new) and
        ``num_frames_per_inference`` (V1 compat) -- the former takes
        precedence.
        """
        nfpc = num_frames_per_chunk or num_frames_per_inference or self.v2_config.num_frames_per_inference
        return self._wrapper.infer_one_step(
            audio_input=audio_input,
            num_frames_per_chunk=nfpc,
            frame_idx=frame_idx,
            gen_text=gen_text,
            audio_toks_buffer=audio_toks_buffer,
            input_embeds_history=input_embeds_history if input_embeds_history is not None else [],
            dynamic_cache=dynamic_cache,
            past_key_values=past_key_values,
            code=code,
            subword_mask=subword_mask,
            gen_asr_text=gen_asr_text,
            request_id=request_id,
            perception_cache=perception_cache,
            has_prompt=has_prompt,
            codec_cache=codec_cache,
        )

    def _compute_pad_audio_sec(self, audio_path: str) -> Optional[float]:
        """Resolve the effective ``pad_audio_to_sec`` value for the wrapper."""
        if self.v2_config.pad_to_duration_secs is not None:
            return float(self.v2_config.pad_to_duration_secs)
        if self.v2_config.silence_padding_sec > 0:
            import librosa

            duration = librosa.get_duration(filename=audio_path)
            return duration + self.v2_config.silence_padding_sec
        return None

    @torch.no_grad()
    def inference_realtime_streaming(
        self,
        audio_path: str,
        num_frames_per_inference: int = None,
        request_id: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run streaming inference on an audio file.

        Delegates entirely to the wrapper's
        ``inference_realtime_streaming``, translating V1-style
        parameters (``silence_padding_sec``) to V2-style
        (``pad_audio_to_sec``).
        """
        nfpc = num_frames_per_inference or self.v2_config.num_frames_per_inference
        pad_to = self._compute_pad_audio_sec(audio_path)
        sys_prompt = system_prompt or self.v2_config.system_prompt

        result = self._wrapper.inference_realtime_streaming(
            audio_path=audio_path,
            num_frames_per_chunk=nfpc,
            request_id=request_id,
            pad_audio_to_sec=pad_to,
            system_prompt=sys_prompt,
        )

        result["input_audio_path"] = audio_path
        total = result.get("tokens_len", torch.tensor([0]))[0].item()
        result.setdefault("debug_info", {"total_frames": total})

        return result

    # ------------------------------------------------------------------
    # Server interface
    # ------------------------------------------------------------------
    def generate(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        if not self._is_loaded:
            return [GenerationResult(error="Model not loaded", request_id=r.request_id) for r in requests]
        if not requests:
            return []

        results = []
        for req in requests:
            start_time = time.time()
            temp_file_path = None

            try:
                audio_path = req.audio_path
                if req.audio_bytes:
                    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    temp_file.write(req.audio_bytes)
                    temp_file.close()
                    temp_file_path = temp_file.name
                    audio_path = temp_file_path

                if not audio_path:
                    results.append(
                        GenerationResult(
                            error="Audio input required for s2s_incremental_v2 backend",
                            request_id=req.request_id,
                        )
                    )
                    continue

                output = self.inference_realtime_streaming(
                    audio_path=audio_path,
                    num_frames_per_inference=self.v2_config.num_frames_per_inference,
                    request_id=req.request_id,
                    system_prompt=req.system_prompt,
                )

                audio_bytes = None
                out_sr = self.target_sample_rate
                if output.get("audio") is not None:
                    import soundfile as sf

                    wav = output["audio"].float().cpu().numpy().squeeze()

                    # Trim to padded input duration (input + extra padding) at target SR.
                    # Matches s2s_voicechat_infer_backend: trim to (input_len + padding) / source_sr * target_sr.
                    # This removes frame-alignment overshoot while keeping the full response.
                    pad_to = self._compute_pad_audio_sec(audio_path)
                    if pad_to is not None:
                        per_sample_pred_len = int(pad_to * out_sr)
                        if 0 < per_sample_pred_len < len(wav):
                            wav = wav[:per_sample_pred_len]

                    max_val = float(np.abs(wav).max()) if wav.size else 0.0
                    if max_val > 0:
                        wav = wav / max_val * 0.95

                    if self.v2_config.merge_user_channel:
                        merged = self._merge_user_model_audio(audio_path, wav, out_sr)
                        buf = io.BytesIO()
                        sf.write(buf, merged, out_sr, format="WAV")
                        audio_bytes = buf.getvalue()
                    else:
                        buf = io.BytesIO()
                        sf.write(buf, wav, out_sr, format="WAV")
                        audio_bytes = buf.getvalue()

                elapsed_ms = (time.time() - start_time) * 1000
                output_text = output["text"][0] if output.get("text") else ""
                asr_text = output["asr_text"][0] if output.get("asr_text") else None
                debug_info = output.get("debug_info", {})

                if self.v2_config.use_asr_as_response and asr_text:
                    cleaned = asr_text
                    cleaned = re.sub(r"<[\$|][^>]*[\$|]>", "", cleaned)
                    cleaned = cleaned.replace("^", "")
                    cleaned = re.sub(r"\s+", " ", cleaned).strip()
                    output_text = cleaned

                request_id_key = req.request_id or datetime.now().strftime("%Y%m%d_%H%M%S")
                artifacts_dir = self._get_artifacts_dir(request_id_key)
                if artifacts_dir:
                    self._save_artifacts(artifacts_dir, audio_path, output_text, audio_bytes, debug_info, elapsed_ms)

                results.append(
                    GenerationResult(
                        text=output_text,
                        asr_text=asr_text,
                        audio_bytes=audio_bytes,
                        audio_sample_rate=out_sr,
                        request_id=req.request_id,
                        generation_time_ms=elapsed_ms,
                        debug_info=debug_info,
                    )
                )

            except Exception as e:
                import traceback

                traceback.print_exc()
                results.append(GenerationResult(error=str(e), request_id=req.request_id))

            finally:
                if temp_file_path and os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)

        return results

    # ------------------------------------------------------------------
    # Validation & health
    # ------------------------------------------------------------------
    def validate_request(self, request: GenerationRequest) -> Optional[str]:
        if not request.audio_bytes and not request.audio_path:
            return "Audio input is required for s2s_incremental_v2 backend"
        return None

    def health_check(self) -> Dict[str, Any]:
        base = super().health_check()
        if self._is_loaded:
            base.update(
                {
                    "buffer_size_frames": self.v2_config.buffer_size_frames,
                    "num_frames_per_inference": self.v2_config.num_frames_per_inference,
                    "decode_audio": self.v2_config.decode_audio,
                    "target_sample_rate": self.target_sample_rate,
                    "tts_enabled": self.generation_config is not None,
                    "engine_type": self.v2_config.engine_type,
                    "use_perception_cache": self.v2_config.use_perception_cache,
                    "use_codec_cache": self.v2_config.use_codec_cache,
                }
            )
        return base

    # ------------------------------------------------------------------
    # Wrapper delegates for session backend compatibility
    # ------------------------------------------------------------------
    def _update_audio_buffer(self, audio_buffer, buffer_fill_level, new_audio, buffer_size_samples):
        return self._wrapper._update_audio_buffer(audio_buffer, buffer_fill_level, new_audio, buffer_size_samples)

    def _clone_cache(self, cache):
        return self._wrapper._clone_cache(cache)

    def _get_bos_embedding(self):
        return self._wrapper._get_bos_embedding()

    def _get_asr_bos_embedding(self):
        return self._wrapper._get_asr_bos_embedding()

    def _samples_per_audio_output_frame(self):
        return self._wrapper._samples_per_audio_output_frame()

    def abort_request(self, request_id: Optional[str] = None) -> bool:
        if self._wrapper is not None:
            return self._wrapper.abort_request(request_id)
        return False

    # ------------------------------------------------------------------
    # Artifact / dual-channel helpers (server-side I/O, not in wrapper)
    # ------------------------------------------------------------------
    def _get_artifacts_dir(self, request_id: str) -> Optional[str]:
        if not self.v2_config.save_session_artifacts:
            return None
        base_dir = self.v2_config.session_artifacts_dir
        artifacts_dir = os.path.join(base_dir, request_id)
        os.makedirs(artifacts_dir, exist_ok=True)
        return artifacts_dir

    def _save_artifacts(
        self,
        artifacts_dir: str,
        input_audio_path: str,
        output_text: str,
        output_audio_bytes: Optional[bytes],
        debug_info: Dict[str, Any],
        generation_time_ms: float,
    ) -> Dict[str, str]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        input_dest = os.path.join(artifacts_dir, f"{timestamp}_input.wav")
        shutil.copy2(input_audio_path, input_dest)

        output_audio_path = None
        if output_audio_bytes:
            output_audio_path = os.path.join(artifacts_dir, f"{timestamp}_output.wav")
            with open(output_audio_path, "wb") as f:
                f.write(output_audio_bytes)

        output_json_path = os.path.join(artifacts_dir, f"{timestamp}_output.json")
        with open(output_json_path, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "text": output_text,
                    "audio_path": output_audio_path,
                    "debug_info": debug_info,
                    "generation_time_ms": generation_time_ms,
                },
                f,
                indent=2,
            )

        return {"artifacts_dir": artifacts_dir, "input_path": input_dest, "output_path": output_audio_path}

    def _merge_user_model_audio(
        self,
        input_audio_path: str,
        pred_audio: np.ndarray,
        pred_sr: int,
    ) -> np.ndarray:
        """Merge user (input) and model (pred) into a two-channel array on one timeline.

        Mirrors s2s_voicechat_infer_backend._merge_user_model_audio:
        channel 0 = user, channel 1 = pred, same length (padded),
        so FDB ASR (--stereo) gets a single timeline.
        """
        import soundfile as sf

        user_audio, user_sr = sf.read(input_audio_path)
        user = np.asarray(user_audio, dtype=np.float64).squeeze()
        pred = np.asarray(pred_audio, dtype=np.float64).squeeze()
        if user.ndim > 1:
            user = user.mean(axis=1)
        if pred.ndim > 1:
            pred = pred.mean(axis=1)
        if user_sr != pred_sr:
            try:
                import librosa
                user = librosa.resample(user, orig_sr=user_sr, target_sr=pred_sr)
            except ImportError:
                import scipy.signal
                user = scipy.signal.resample(user, int(len(user) * pred_sr / user_sr))
        T1, T2 = user.shape[0], pred.shape[0]
        max_len = max(T1, T2)
        user_pad = np.pad(user, (0, max_len - T1), mode="constant", constant_values=0)
        pred_pad = np.pad(pred, (0, max_len - T2), mode="constant", constant_values=0)
        merged = np.stack([user_pad, pred_pad], axis=1).astype(np.float32)
        return merged

    # ------------------------------------------------------------------
    # Frame alignment utilities (kept for debug/session compat)
    # ------------------------------------------------------------------
    def _decode_single_token(self, token_id: int, pad_id: int) -> str:
        try:
            tokens = self._tokenizer.ids_to_tokens([token_id])
            if tokens:
                return tokens[0].replace("\u0120", " ")
            return f"<SPEC_{token_id}>"
        except Exception:
            return f"<SPEC_{token_id}>"

    def _init_frame_alignment(self) -> Dict[str, list]:
        return {
            "frame_idx": [],
            "user_stream": [],
            "agent_stream_token": [],
            "agent_stream_decoded": [],
            "asr_stream_token": [],
            "asr_stream_decoded": [],
            "is_tts_stop": [],
        }

    def _append_frame_alignment(
        self,
        frame_alignment: Dict[str, list],
        frame_idx: int,
        phase: str,
        gen_text: torch.Tensor,
        gen_asr_text: torch.Tensor,
        pad_id: int,
        is_tts_stop: bool = False,
    ) -> None:
        agent_token = gen_text[0, frame_idx].item() if frame_idx < gen_text.shape[1] else pad_id
        asr_token = gen_asr_text[0, frame_idx].item() if frame_idx < gen_asr_text.shape[1] else pad_id
        frame_alignment["frame_idx"].append(frame_idx)
        frame_alignment["user_stream"].append(phase)
        frame_alignment["agent_stream_token"].append(agent_token)
        frame_alignment["agent_stream_decoded"].append(self._decode_single_token(agent_token, pad_id))
        frame_alignment["asr_stream_token"].append(asr_token)
        frame_alignment["asr_stream_decoded"].append(self._decode_single_token(asr_token, pad_id))
        frame_alignment["is_tts_stop"].append(is_tts_stop)
