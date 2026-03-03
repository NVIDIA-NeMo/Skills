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
Backend implementations for the Unified NeMo Inference Server.

Available backends:
- magpie_tts: MagpieTTS text-to-speech (audio output from text input)
- nemo_asr: NeMo ASR speech-to-text (text output from audio input)

Backends are lazily loaded to avoid importing heavy dependencies upfront.
"""

import importlib

from .base import BackendConfig, GenerationRequest, GenerationResult, InferenceBackend, Modality

__all__ = [
    "InferenceBackend",
    "GenerationRequest",
    "GenerationResult",
    "BackendConfig",
    "Modality",
    "get_backend",
    "list_backends",
]

# Registry of available backends: name -> (module_name, class_name)
BACKEND_REGISTRY = {
    "magpie_tts": ("magpie_tts_backend", "MagpieTTSBackend"),
    "nemo_asr": ("nemo_asr_backend", "NeMoASRBackend"),
}


def list_backends() -> list:
    """Return list of available backend names."""
    return list(BACKEND_REGISTRY.keys())


def get_backend(backend_name: str) -> type:
    """Get backend class by name with lazy loading.

    Args:
        backend_name: One of the registered backend names

    Returns:
        Backend class (not instance)

    Raises:
        ValueError: If backend name is unknown
        ImportError: If backend dependencies are not available
    """
    if backend_name not in BACKEND_REGISTRY:
        available = ", ".join(BACKEND_REGISTRY.keys())
        raise ValueError(f"Unknown backend: '{backend_name}'. Available backends: {available}")

    module_name, class_name = BACKEND_REGISTRY[backend_name]

    try:
        module = importlib.import_module(f".{module_name}", package=__name__)
        return getattr(module, class_name)
    except ImportError as e:
        raise ImportError(
            f"Failed to import backend '{backend_name}'. Make sure required dependencies are installed. Error: {e}"
        ) from e
