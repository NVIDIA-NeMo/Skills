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
Abstract base class for inference backends.

All model backends must implement this interface to be usable with the
unified inference server.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


class Modality(str, Enum):
    """Supported input/output modalities."""

    TEXT = "text"
    AUDIO_IN = "audio_in"
    AUDIO_OUT = "audio_out"


@dataclass
class BackendConfig:
    """Base configuration for all backends.

    Subclasses should add their own fields. The from_dict() classmethod
    handles extracting known fields and putting the rest in extra_config.
    """

    model_path: str = ""
    device: str = "cuda"
    dtype: str = "bfloat16"

    # Generation defaults
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: Optional[int] = None

    # Additional model-specific configs passed through
    extra_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BackendConfig":
        """Create config from dictionary, extracting known fields."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        known = {k: v for k, v in d.items() if k in known_fields and k != "extra_config"}
        extra = {k: v for k, v in d.items() if k not in known_fields}
        return cls(**known, extra_config=extra)


@dataclass
class GenerationRequest:
    """A single generation request.

    Supports text and/or audio inputs depending on the backend's capabilities.
    """

    # Text inputs
    text: Optional[str] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None

    # Audio input (raw bytes)
    audio_bytes: Optional[bytes] = None
    sample_rate: int = 16000

    # Multi-turn audio inputs (list of raw audio bytes)
    audio_bytes_list: Optional[List[bytes]] = None

    # Audio path reference (file path, tar member, or datastore URI)
    audio_path: Optional[str] = None

    # Generation parameters (override backend defaults)
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    seed: Optional[int] = None

    # Additional parameters
    extra_params: Dict[str, Any] = field(default_factory=dict)

    # Request tracking
    request_id: Optional[str] = None


@dataclass
class GenerationResult:
    """Result from a generation request.

    Contains text output and optionally audio output, plus metadata.
    """

    # Text output
    text: str = ""

    # Audio output (raw bytes, can be encoded to base64 for JSON)
    audio_bytes: Optional[bytes] = None
    audio_sample_rate: int = 16000
    audio_format: str = "wav"

    # Metadata
    request_id: Optional[str] = None
    num_tokens_generated: int = 0
    generation_time_ms: float = 0.0

    # Debug info (optional, backend-specific)
    debug_info: Optional[Dict[str, Any]] = None

    # Error handling
    error: Optional[str] = None

    def is_success(self) -> bool:
        return self.error is None


class InferenceBackend(ABC):
    """Abstract base class for inference backends.

    Implementations must provide:
    - get_config_class(): Return the config dataclass for this backend
    - load_model(): Initialize the model from config
    - generate(): Run inference on a batch of requests
    - supported_modalities: What input/output types are supported

    Optionally:
    - get_extra_routes(): Return additional FastAPI routes this backend needs
      (e.g., session management endpoints for S2S backends)
    """

    def __init__(self, config: BackendConfig):
        self.config = config
        self._model = None
        self._is_loaded = False

    @classmethod
    @abstractmethod
    def get_config_class(cls) -> type:
        """Return the config dataclass for this backend.

        The returned class must be a subclass of BackendConfig with a
        from_dict() classmethod.
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name (e.g., 'salm', 'magpie_tts')."""
        ...

    @property
    @abstractmethod
    def supported_modalities(self) -> Set[Modality]:
        """Return the set of supported modalities."""
        ...

    @abstractmethod
    def load_model(self) -> None:
        """Load and initialize the model.

        Should set self._model and self._is_loaded = True on success.
        Called once during server startup.
        """
        ...

    @abstractmethod
    def generate(self, requests: List[GenerationRequest]) -> List[GenerationResult]:
        """Run inference on a batch of requests.

        Args:
            requests: List of generation requests to process

        Returns:
            List of generation results, one per request (same order)
        """
        ...

    @classmethod
    def get_extra_routes(cls, backend_instance: "InferenceBackend") -> list:
        """Return additional FastAPI routes this backend needs.

        Override to register custom endpoints (e.g., session management).
        Each item should be a dict with 'path', 'endpoint', 'methods'.

        Returns:
            List of route dicts, empty by default.
        """
        return []

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._is_loaded

    def health_check(self) -> Dict[str, Any]:
        """Return health status information."""
        return {
            "backend": self.name,
            "model_loaded": self._is_loaded,
            "model_path": self.config.model_path,
            "device": self.config.device,
            "modalities": [m.value for m in self.supported_modalities],
        }

    def get_generation_params(self, request: GenerationRequest) -> Dict[str, Any]:
        """Get effective generation parameters, merging request with config defaults."""
        return {
            "max_new_tokens": request.max_new_tokens
            if request.max_new_tokens is not None
            else self.config.max_new_tokens,
            "temperature": request.temperature if request.temperature is not None else self.config.temperature,
            "top_p": request.top_p if request.top_p is not None else self.config.top_p,
            "top_k": request.top_k if request.top_k is not None else self.config.top_k,
        }

    def validate_request(self, request: GenerationRequest) -> Optional[str]:
        """Validate a request against supported modalities.

        Returns:
            Error message if invalid, None if valid
        """
        modalities = self.supported_modalities

        has_text_input = request.text is not None
        has_audio_input = request.audio_bytes is not None or (
            request.audio_bytes_list is not None and len(request.audio_bytes_list) > 0
        )

        if has_audio_input and Modality.AUDIO_IN not in modalities:
            return f"Backend '{self.name}' does not support audio input"

        if not has_text_input and not has_audio_input:
            return "Request must have either text or audio input"

        return None
