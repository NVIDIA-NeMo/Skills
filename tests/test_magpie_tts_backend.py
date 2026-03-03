# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from pathlib import Path

import pytest

from recipes.multimodal.server.backends.magpie_tts_backend import MagpieTTSBackend, MagpieTTSConfig


def test_context_audio_path_is_disabled_without_allowlist(tmp_path: Path):
    backend = MagpieTTSBackend(
        MagpieTTSConfig(model_path="dummy", codec_model_path="dummy", context_audio_allowed_roots=None)
    )
    target = tmp_path / "ctx.wav"
    target.write_bytes(b"fake")

    with pytest.raises(ValueError, match="context_audio_filepath is disabled"):
        backend._resolve_context_audio_path(str(target))


def test_context_audio_path_must_be_under_allowed_roots(tmp_path: Path):
    allowed = tmp_path / "allowed"
    outside = tmp_path / "outside"
    allowed.mkdir()
    outside.mkdir()
    in_root = allowed / "in_root.wav"
    in_root.write_bytes(b"fake")
    out_of_root = outside / "out_of_root.wav"
    out_of_root.write_bytes(b"fake")

    backend = MagpieTTSBackend(
        MagpieTTSConfig(
            model_path="dummy",
            codec_model_path="dummy",
            context_audio_allowed_roots=[str(allowed)],
        )
    )

    assert backend._resolve_context_audio_path(str(in_root)) == str(in_root.resolve())
    with pytest.raises(ValueError, match="outside allowed roots"):
        backend._resolve_context_audio_path(str(out_of_root))
