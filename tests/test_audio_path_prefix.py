# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import importlib
import sys
import types
from pathlib import Path


def _stub_audio_prepare_deps(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "latex2sympy2_extended",
        types.SimpleNamespace(NormalizationConfig=object, normalize_latex=lambda value, **kwargs: value),
    )
    monkeypatch.setitem(
        sys.modules,
        "math_verify",
        types.SimpleNamespace(
            LatexExtractionConfig=object,
            StringExtractionConfig=object,
            parse=lambda *args, **kwargs: [],
            verify=lambda *args, **kwargs: False,
        ),
    )
    soundfile = types.SimpleNamespace(
        write=lambda *args, **kwargs: None,
        info=lambda *args, **kwargs: types.SimpleNamespace(frames=16000, samplerate=16000),
    )
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=lambda *args, **kwargs: []))
    monkeypatch.setitem(sys.modules, "tqdm", types.SimpleNamespace(tqdm=lambda iterable, **kwargs: iterable))


def test_container_audio_path_helpers(monkeypatch):
    _stub_audio_prepare_deps(monkeypatch)
    from nemo_skills.dataset.utils import build_container_audio_path, get_container_audio_root

    assert get_container_audio_root("/dataset/") == "/dataset"
    monkeypatch.setenv("NEMO_SKILLS_AUDIO_ROOT", "/mnt/audio")
    assert get_container_audio_root() == "/mnt/audio"
    assert (
        build_container_audio_path("asr-leaderboard", "data", "sample.flac", audio_prefix="/data/")
        == "/data/asr-leaderboard/data/sample.flac"
    )


def test_asr_leaderboard_audio_prefix(monkeypatch, tmp_path):
    _stub_audio_prepare_deps(monkeypatch)
    prepare = importlib.import_module("nemo_skills.dataset.asr-leaderboard.prepare")
    entry = {
        "id": "sample/001",
        "text": "hello world",
        "audio": {"array": [0.0] * 1600, "sampling_rate": 16000},
    }

    formatted = prepare.save_audio_and_format_entry(
        entry,
        dataset_name="librispeech_clean",
        audio_dir=tmp_path,
        sample_idx=0,
        with_audio=False,
        audio_root="/data",
    )

    assert (
        formatted["messages"][1]["audio"]["path"]
        == "/data/asr-leaderboard/data/librispeech_clean/sample_001.flac"
    )


def test_fleurs_audio_prefix(monkeypatch, tmp_path):
    _stub_audio_prepare_deps(monkeypatch)
    module_path = tmp_path / "fleurs.py"
    module_path.write_text(
        "_FLEURS_LANG = {'en_us'}\n"
        "_FLEURS_LANG_TO_LONG = {'en_us': 'English'}\n"
        "_FLEURS_LANG_TO_GROUP = {'en_us': 'western'}\n"
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(hf_hub_download=lambda **kwargs: str(module_path)),
    )
    sys.modules.pop("nemo_skills.dataset.fleurs.prepare", None)
    prepare = importlib.import_module("nemo_skills.dataset.fleurs.prepare")

    assert prepare.get_container_audio_path("en_us", "audio.wav", audio_root="/data") == (
        "/data/fleurs/audio/en_us/audio.wav"
    )


def test_covost2_audio_prefix(monkeypatch):
    _stub_audio_prepare_deps(monkeypatch)
    prepare = importlib.import_module("nemo_skills.dataset.covost2.prepare")

    assert prepare.get_container_audio_path("fr", "test", "common_voice_fr_123", audio_root="/data") == (
        "/data/covost2/audio/fr/test/common_voice_fr_123.wav"
    )


def test_librispeech_pc_audio_prefix(monkeypatch, tmp_path):
    _stub_audio_prepare_deps(monkeypatch)
    prepare = importlib.import_module("nemo_skills.dataset.librispeech-pc.prepare")
    manifest = tmp_path / "test-clean.json"
    manifest.write_text(
        '{"audio_filepath": "LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac", "text": "Hello world."}\n'
    )

    count = prepare.process_split("test-clean", tmp_path, tmp_path, with_audio=False, audio_root="/data")

    assert count == 1
    row = (tmp_path / "test-clean.jsonl").read_text()
    assert "/data/librispeech-pc/LibriSpeech/test-clean/1089/134686/1089-134686-0000.flac" in row


def test_contextasr_audio_prefix(monkeypatch):
    _stub_audio_prepare_deps(monkeypatch)
    prepare = importlib.import_module("nemo_skills.dataset.contextasr-bench.prepare")
    sample = {
        "audio": "audio/ContextASR-Speech/English/sample.wav",
        "duration": 1.25,
        "entity_list": ["Aspirin"],
        "domain_label": "Medical",
        "text": "take aspirin daily",
        "uniq_id": "sample",
    }

    assert prepare.resolve_audio_prefix("/data") == "/data/contextasr-bench"
    monkeypatch.setenv("NEMO_SKILLS_AUDIO_ROOT", "/mnt/audio")
    assert prepare.resolve_audio_prefix() == "/mnt/audio/contextasr-bench"

    formatted = prepare.format_entry(sample, "fine", "/data/contextasr-bench")

    assert formatted["audio_filepath"] == "/data/contextasr-bench/audio/ContextASR-Speech/English/sample.wav"
    assert formatted["messages"][0]["audio"]["path"] == (
        "/data/contextasr-bench/audio/ContextASR-Speech/English/sample.wav"
    )
