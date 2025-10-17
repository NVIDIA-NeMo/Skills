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

"""Utilities for NIM model integration."""

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

LOG = logging.getLogger(__name__)


@dataclass
class TTSExtraConfig:
    """Configuration for TTS NIM generation parameters.

    These parameters can be passed via the extra_body argument to override
    the model defaults.
    """

    # Voice and language settings
    voice: Optional[str] = None
    language_code: Optional[str] = None
    sample_rate_hz: Optional[int] = None

    # Zero-shot voice cloning
    zero_shot_audio_prompt_file: Optional[str] = None
    zero_shot_quality: int = 20
    zero_shot_transcript: Optional[str] = None

    # Advanced settings
    custom_dictionary: Dict[str, str] = field(default_factory=dict)
    output_filename: Optional[str] = None


@dataclass
class ASRExtraConfig:
    """Configuration for ASR NIM generation parameters.

    These parameters can be passed via the extra_body argument to override
    the model defaults.
    """

    # Basic recognition settings
    language_code: Optional[str] = None
    max_alternatives: int = 1
    profanity_filter: bool = False
    automatic_punctuation: bool = True
    no_verbatim_transcripts: bool = False
    word_time_offsets: bool = False

    # Word boosting
    boosted_lm_words: Optional[List[str]] = None
    boosted_lm_score: float = 15.0

    # Speaker diarization
    speaker_diarization: bool = False
    diarization_max_speakers: int = 2

    # Endpoint detection
    start_history: int = 0
    start_threshold: float = 0.0
    stop_history: int = 0
    stop_history_eou: int = 0
    stop_threshold: float = 0.0
    stop_threshold_eou: float = 0.0

    # Custom configuration
    custom_configuration: str = ""


def setup_ssh_tunnel(
    server_host: str,
    server_port: str,
    ssh_server: Optional[str] = None,
    ssh_key_path: Optional[str] = None,
) -> Tuple[str, str, Optional[object]]:
    """Setup SSH tunnel if ssh_server and ssh_key_path are provided.

    Args:
        server_host: Host of the server
        server_port: Port of the server
        ssh_server: SSH server for tunneling (format: [user@]host)
        ssh_key_path: Path to SSH key

    Returns:
        Tuple of (final_host, final_port, tunnel_object)
        If no tunnel is created, returns (server_host, server_port, None)
    """
    # Check environment variables if not provided
    if ssh_server is None:
        ssh_server = os.getenv("NEMO_SKILLS_SSH_SERVER")
    if ssh_key_path is None:
        ssh_key_path = os.getenv("NEMO_SKILLS_SSH_KEY_PATH")

    # No tunnel needed
    if not ssh_server or not ssh_key_path:
        return server_host, server_port, None

    # Setup tunnel
    try:
        import sshtunnel
    except ImportError:
        LOG.warning("sshtunnel not installed, cannot create SSH tunnel")
        return server_host, server_port, None

    # Parse username from ssh_server if present
    if "@" in ssh_server:
        ssh_username, ssh_server = ssh_server.split("@")
    else:
        ssh_username = None

    tunnel = sshtunnel.SSHTunnelForwarder(
        (ssh_server, 22),
        ssh_username=ssh_username,
        ssh_pkey=ssh_key_path,
        remote_bind_address=(server_host, int(server_port)),
    )
    tunnel.start()

    final_host = "127.0.0.1"
    final_port = str(tunnel.local_bind_port)

    LOG.info(f"SSH tunnel created: {ssh_server} -> {server_host}:{server_port} mapped to localhost:{final_port}")

    return final_host, final_port, tunnel


def validate_unsupported_params(kwargs: dict, model_name: str = "NIM model") -> None:
    """Validate that unsupported LLM parameters are not set to non-default values.

    Args:
        kwargs: Keyword arguments passed to generate
        model_name: Name of the model for error messages

    Raises:
        ValueError: If unsupported parameters are set to non-default values
    """
    unsupported_checks = {
        "temperature": (kwargs.get("temperature", 0.0), 0.0),
        "top_p": (kwargs.get("top_p", 0.95), 0.95),
        "top_k": (kwargs.get("top_k", 0), 0),
        "min_p": (kwargs.get("min_p", 0.0), 0.0),
        "repetition_penalty": (kwargs.get("repetition_penalty", 1.0), 1.0),
        "tokens_to_generate": (kwargs.get("tokens_to_generate", 2048), 2048),
        "stop_phrases": (kwargs.get("stop_phrases"), None),
        "tools": (kwargs.get("tools"), None),
        "reasoning_effort": (kwargs.get("reasoning_effort"), None),
    }

    non_default = []
    for param, (val, default) in unsupported_checks.items():
        if val is not None and val != default:
            non_default.append(f"{param}={val}")

    if non_default:
        LOG.warning(
            f"{model_name} does not support LLM parameters: {', '.join(non_default)}. "
            f"These parameters are ignored. Use extra_body for model-specific options."
        )
