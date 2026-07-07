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

import os
import tarfile
import tempfile
from pathlib import Path
from unittest.mock import patch
from urllib.parse import urlparse

from recipes.multimodal.server.backends import GenerationRequest
from recipes.multimodal.server.backends.nemo_asr_backend import NeMoASRBackend, NeMoASRConfig


def _is_datastore_path_fallback(path: str) -> bool:
    """Pure-Python fallback used by tests so they never import nemo.utils."""
    parsed = urlparse(path)
    return parsed.scheme in ("ais", "s3", "gs", "hdfs") and bool(parsed.netloc)


class _FakeHypothesis:
    def __init__(self, text: str):
        self.text = text
        self.words = [{"word": text, "start_time": 0.0, "end_time": 0.2, "confidence": 1.0}]


class _FakeTimestampHypothesis:
    def __init__(self):
        self.text = "hello world"
        self.words = ["hello", "world"]
        self.timestamp = {
            "word": [
                {"word": "hello", "start": 0.1, "end": 0.5},
                {"word": "world", "start": 0.5, "end": 0.9},
            ]
        }


class _FakeASRModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, audio=None, **kwargs):
        self.calls.append((audio, kwargs))
        return [_FakeHypothesis(f"transcript_{idx}") for idx, _ in enumerate(audio)]


def test_nemo_asr_backend_validate_request_requires_audio():
    backend = NeMoASRBackend(NeMoASRConfig(model_path="dummy"))
    err = backend.validate_request(GenerationRequest(text="x"))
    assert err is not None


def test_generation_params_preserve_explicit_zero_values():
    backend = NeMoASRBackend(
        NeMoASRConfig(model_path="dummy", max_new_tokens=128, temperature=0.8, top_p=0.95, top_k=40)
    )

    params = backend.get_generation_params(GenerationRequest(temperature=0.0, top_p=0.0, top_k=0))

    assert params["max_new_tokens"] == 128
    assert params["temperature"] == 0.0
    assert params["top_p"] == 0.0
    assert params["top_k"] == 0


def test_nemo_asr_backend_generate_batched_with_words():
    backend = NeMoASRBackend(NeMoASRConfig(model_path="dummy", batch_size=4))
    backend._model = _FakeASRModel()
    backend._is_loaded = True

    reqs = [
        GenerationRequest(
            audio_bytes=b"RIFF" + b"\x00" * 64, request_id="r1", extra_params={"return_hypotheses": True}
        ),
        GenerationRequest(
            audio_bytes=b"RIFF" + b"\x00" * 64, request_id="r2", extra_params={"return_hypotheses": True}
        ),
    ]
    results = backend.generate(reqs)

    assert len(results) == 2
    assert results[0].text == "transcript_0"
    assert results[1].text == "transcript_1"
    assert results[0].debug_info["words"][0]["word"] == "transcript_0"
    assert results[1].debug_info["words"][0]["word"] == "transcript_1"
    assert results[0].request_id == "r1"
    assert results[1].request_id == "r2"


def test_nemo_asr_backend_prefers_timestamp_words_when_words_are_strings():
    backend = NeMoASRBackend(NeMoASRConfig(model_path="dummy"))
    text, words = backend._parse_single_hypothesis(_FakeTimestampHypothesis())

    assert text == "hello world"
    assert words == [
        {"word": "hello", "start_time": 0.1, "end_time": 0.5, "confidence": None},
        {"word": "world", "start_time": 0.5, "end_time": 0.9, "confidence": None},
    ]


# ---------------------------------------------------------------------------
# Tarred audio / datastore resolution tests
# ---------------------------------------------------------------------------


def _make_tar_with_member(tar_path: str, member_name: str, content: bytes) -> None:
    """Helper: create a tar archive containing a single member."""
    import io

    with tarfile.open(tar_path, "w") as tar:
        info = tarfile.TarInfo(name=member_name)
        info.size = len(content)
        tar.addfile(info, io.BytesIO(content))


_PATCH_IS_DS = patch.object(NeMoASRBackend, "_is_datastore_path", staticmethod(_is_datastore_path_fallback))


def test_is_datastore_path_detects_ais_and_s3():
    with _PATCH_IS_DS:
        assert NeMoASRBackend._is_datastore_path("ais://bucket/audio.tar") is True
        assert NeMoASRBackend._is_datastore_path("s3://bucket/audio.tar") is True
        assert NeMoASRBackend._is_datastore_path("/local/path/audio.tar") is False
        assert NeMoASRBackend._is_datastore_path("relative/audio.tar") is False


def test_resolve_tarred_audio_files_with_local_tar():
    """_resolve_tarred_audio_files resolves a glob to matching .tar files."""
    with _PATCH_IS_DS, tempfile.TemporaryDirectory() as tmpdir:
        tar1 = os.path.join(tmpdir, "audio_0.tar")
        tar2 = os.path.join(tmpdir, "audio_1.tar")
        _make_tar_with_member(tar1, "a.wav", b"audio_a")
        _make_tar_with_member(tar2, "b.wav", b"audio_b")

        backend = NeMoASRBackend(
            NeMoASRConfig(
                model_path="dummy",
                tarred_audio_filepaths=os.path.join(tmpdir, "audio_*.tar"),
            )
        )

        assert len(backend._tarred_audio_files) == 2
        assert str(Path(tar1).absolute()) in backend._tarred_audio_files
        assert str(Path(tar2).absolute()) in backend._tarred_audio_files


def test_resolve_audio_path_tar_member():
    """Implicit tar member lookup finds audio inside configured shards."""
    audio_content = b"RIFF_fake_audio_data"
    with _PATCH_IS_DS, tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, "audio_0.tar")
        _make_tar_with_member(tar_path, "sample.wav", audio_content)

        backend = NeMoASRBackend(NeMoASRConfig(model_path="dummy", tarred_audio_filepaths=tar_path))

        resolved, cleanup = backend._resolve_audio_path("sample.wav")
        try:
            assert resolved.read_bytes() == audio_content
            assert cleanup is not None  # extracted to temp file
        finally:
            if cleanup is not None:
                cleanup.unlink(missing_ok=True)


def test_resolve_request_audio_with_shard_id():
    """Shard ID optimisation jumps to matching audio_{shard_id}.tar."""
    audio_content = b"RIFF_shard_audio"
    with _PATCH_IS_DS, tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, "audio_9.tar")
        _make_tar_with_member(tar_path, "clip.wav", audio_content)

        backend = NeMoASRBackend(NeMoASRConfig(model_path="dummy", tarred_audio_filepaths=tar_path))

        req = GenerationRequest(
            audio_path="clip.wav",
            extra_params={"shard_id": 9},
        )
        audio_bytes, cleanup = backend._resolve_request_audio(req)
        try:
            assert audio_bytes == audio_content
        finally:
            if cleanup is not None:
                cleanup.unlink(missing_ok=True)


def test_resolve_audio_path_explicit_tar_member_syntax():
    """Explicit ``tar_path::member`` syntax extracts the right member."""
    audio_content = b"RIFF_explicit_member"
    with _PATCH_IS_DS, tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, "shard.tar")
        _make_tar_with_member(tar_path, "utterance.flac", audio_content)

        backend = NeMoASRBackend(NeMoASRConfig(model_path="dummy"))

        audio_ref = f"{tar_path}::utterance.flac"
        resolved, cleanup = backend._resolve_audio_path(audio_ref)
        try:
            assert resolved.read_bytes() == audio_content
            assert cleanup is not None
        finally:
            if cleanup is not None:
                cleanup.unlink(missing_ok=True)


def test_validate_request_accepts_extra_params_audio_filepath():
    """validate_request passes when audio_filepath is in extra_params."""
    backend = NeMoASRBackend(NeMoASRConfig(model_path="dummy"))
    err = backend.validate_request(GenerationRequest(extra_params={"audio_filepath": "/data/audio.wav"}))
    assert err is None


def test_generate_with_tar_audio():
    """End-to-end generate using audio from a tar archive."""
    audio_content = b"RIFF" + b"\x00" * 64
    with _PATCH_IS_DS, tempfile.TemporaryDirectory() as tmpdir:
        tar_path = os.path.join(tmpdir, "audio_0.tar")
        _make_tar_with_member(tar_path, "utt.wav", audio_content)

        backend = NeMoASRBackend(NeMoASRConfig(model_path="dummy", tarred_audio_filepaths=tar_path))
        backend._model = _FakeASRModel()
        backend._is_loaded = True

        reqs = [
            GenerationRequest(
                audio_path="utt.wav",
                request_id="tar_r1",
                extra_params={"return_hypotheses": True},
            )
        ]
        results = backend.generate(reqs)

        assert len(results) == 1
        assert results[0].text == "transcript_0"
        assert results[0].request_id == "tar_r1"
        assert results[0].error is None
