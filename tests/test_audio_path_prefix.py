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
import json
import sys
import types
from pathlib import Path

AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".opus")


def _walk_strings(value):
    """Yield every string leaf in a nested JSON-style value."""
    if isinstance(value, dict):
        for v in value.values():
            yield from _walk_strings(v)
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, str):
        yield value


def _assert_clean_audio_paths(payload, expected_prefix):
    """Assert no stale `/dataset/` paths and every audio path uses the expected prefix.

    `payload` may be a dict (single record), a list of dicts, or a `Path` to a JSONL file.
    """
    if isinstance(payload, Path):
        records = [json.loads(line) for line in payload.read_text().splitlines() if line.strip()]
    elif isinstance(payload, list):
        records = payload
    else:
        records = [payload]

    for idx, record in enumerate(records):
        for s in _walk_strings(record):
            assert "/dataset/" not in s, f"record {idx}: stale '/dataset/' substring in {s!r}"
            if s.endswith(AUDIO_EXTS):
                assert s.startswith(expected_prefix), (
                    f"record {idx}: audio path {s!r} does not start with {expected_prefix!r}"
                )


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
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=lambda *args, **kwargs: [], Audio=lambda *args, **kwargs: None),
    )
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

    formatted = prepare.format_entry(
        entry,
        "librispeech_clean",
        tmp_path,
        text_field="text",
        id_field="id",
        with_audio=False,
        audio_root="/data",
    )

    assert formatted["messages"][1]["audio"]["path"] == "/data/asr-leaderboard/data/librispeech_clean/sample_001.flac"
    _assert_clean_audio_paths(formatted, "/data/asr-leaderboard/")


def test_asr_leaderboard_end_to_end(monkeypatch, tmp_path):
    """Run prepare_dataset against a stubbed HF source and assert the full JSONL is clean."""
    _stub_audio_prepare_deps(monkeypatch)
    prepare = importlib.import_module("nemo_skills.dataset.asr-leaderboard.prepare")

    fake_dataset = [
        {
            "id": "sample/001",
            "text": "hello world",
            "audio": {"array": [0.0] * 1600, "sampling_rate": 16000},
        },
        {
            "id": "sample/002",
            "text": "foo bar",
            "audio": {"array": [0.0] * 1600, "sampling_rate": 16000},
        },
    ]
    monkeypatch.setattr(prepare, "load_dataset", lambda *args, **kwargs: fake_dataset)

    count = prepare.prepare_dataset("librispeech_clean", tmp_path, with_audio=False, audio_root="/data")

    assert count == 2
    _assert_clean_audio_paths(tmp_path / "librispeech_clean.jsonl", "/data/asr-leaderboard/")


def _load_fleurs_prepare(monkeypatch, tmp_path):
    """Stub HF download for the FLEURS metadata module and return the prepare module."""
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
    return importlib.import_module("nemo_skills.dataset.fleurs.prepare")


def test_fleurs_audio_prefix(monkeypatch, tmp_path):
    _stub_audio_prepare_deps(monkeypatch)
    prepare = _load_fleurs_prepare(monkeypatch, tmp_path)

    assert prepare.get_container_audio_path("en_us", "audio.wav", audio_root="/data") == (
        "/data/fleurs/audio/en_us/audio.wav"
    )


def test_fleurs_end_to_end(monkeypatch, tmp_path):
    """Run prepare_fleurs against a stubbed loader and assert the full JSONL is clean."""
    _stub_audio_prepare_deps(monkeypatch)
    prepare = _load_fleurs_prepare(monkeypatch, tmp_path)

    fake_rows = [
        {
            "id": 1,
            "wav_filename": "001.wav",
            "transcription": "hello world",
            "raw_transcription": "Hello, world.",
            "audio": {"array": [0.0] * 1600, "sampling_rate": 16000},
        },
        {
            "id": 2,
            "wav_filename": "002.wav",
            "transcription": "foo bar",
            "raw_transcription": "Foo bar.",
            "audio": {"array": [0.0] * 1600, "sampling_rate": 16000},
        },
    ]
    monkeypatch.setattr(prepare, "load_fleurs", lambda locale, split, local_dir: list(fake_rows))

    data_dir = tmp_path / "fleurs"
    data_dir.mkdir(parents=True, exist_ok=True)
    prepare.prepare_fleurs(
        data_dir=data_dir,
        split="test",
        languages=["en_us"],
        no_audio=True,
        audio_root="/data",
    )

    _assert_clean_audio_paths(data_dir / "asr" / "test.jsonl", "/data/fleurs/")


def test_covost2_audio_prefix(monkeypatch):
    _stub_audio_prepare_deps(monkeypatch)
    prepare = importlib.import_module("nemo_skills.dataset.covost2.prepare")

    assert prepare.get_container_audio_path("fr", "test", "common_voice_fr_123", audio_root="/data") == (
        "/data/covost2/audio/fr/test/common_voice_fr_123.wav"
    )


def test_covost2_end_to_end(monkeypatch, tmp_path):
    """Run prepare_covost2 ASR with fixture WAVs and TSV; assert the full JSONL is clean."""
    _stub_audio_prepare_deps(monkeypatch)
    prepare = importlib.import_module("nemo_skills.dataset.covost2.prepare")

    cv_data_dir = tmp_path / "cv"
    audio_split_dir = cv_data_dir / "fr" / "test"
    audio_split_dir.mkdir(parents=True, exist_ok=True)
    (audio_split_dir / "common_voice_fr_001.wav").touch()
    (audio_split_dir / "common_voice_fr_002.wav").touch()

    validated_tsv = tmp_path / "validated.tsv"
    validated_tsv.write_text(
        "path\tsplit\tlang\tsentence\n"
        "common_voice_fr_001.wav\ttest\tfr\tbonjour le monde\n"
        "common_voice_fr_002.wav\ttest\tfr\tau revoir\n"
    )

    data_dir = tmp_path / "out"
    data_dir.mkdir(parents=True, exist_ok=True)

    # main's prepare_covost2 produces both ASR + ST; stub the ST loader so the
    # test stays offline (ST would otherwise download the CoVoST TSV).
    monkeypatch.setattr(prepare, "load_covost2", lambda *args, **kwargs: [])
    prepare.prepare_covost2(
        data_dir=data_dir,
        split="test",
        languages=["fr"],
        cv_data_dir=cv_data_dir,
        validated_tsv=validated_tsv,
        audio_root="/data",
    )

    _assert_clean_audio_paths(data_dir / "asr" / "test.jsonl", "/data/covost2/")


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
    _assert_clean_audio_paths(tmp_path / "test-clean.jsonl", "/data/librispeech-pc/")


def test_audiobench_audio_prefix(monkeypatch):
    _stub_audio_prepare_deps(monkeypatch)
    prepare = importlib.import_module("nemo_skills.dataset.audiobench.prepare")
    entry = {
        "instruction": "Transcribe this.",
        "reference": "hello",
        "task_type": "asr",
    }

    formatted = prepare.create_manifest_entry(
        sample=entry,
        audio_filename="sample.wav",
        duration=1.0,
        dataset_name="librispeech_test_clean",
        sample_id=0,
        category="nonjudge",
        audio_root="/data",
    )

    assert formatted["audio_path"] == ["/data/audiobench/nonjudge/audio/librispeech_test_clean/sample.wav"]
    assert formatted["messages"][1]["audio"]["path"] == (
        "/data/audiobench/nonjudge/audio/librispeech_test_clean/sample.wav"
    )
    _assert_clean_audio_paths(formatted, "/data/audiobench/")


def test_musan_audio_prefix(monkeypatch):
    _stub_audio_prepare_deps(monkeypatch)
    prepare = importlib.import_module("nemo_skills.dataset.musan.prepare")

    formatted = prepare.create_manifest_entry(
        audio_filename="noise.wav",
        duration=1.0,
        category="noise",
        sample_id=0,
        label="noise",
        audio_root="/data",
    )

    assert formatted["audio_path"] == ["/data/musan/noise/audio/noise.wav"]
    assert formatted["messages"][1]["audio"]["path"] == "/data/musan/noise/audio/noise.wav"
    _assert_clean_audio_paths(formatted, "/data/musan/")


def test_numb3rs_audio_prefix(monkeypatch, tmp_path):
    _stub_audio_prepare_deps(monkeypatch)
    prepare = importlib.import_module("nemo_skills.dataset.numb3rs.prepare")
    entry = {
        "original_text": "$12.00",
        "text": "twelve dollars",
        "file_name": "MONEY/MONEY_001.wav",
        "duration": 1.0,
        "audio": {"array": [0.0] * 1600, "sampling_rate": 16000},
    }

    formatted = prepare.save_audio_and_format_entry(
        entry,
        category="MONEY",
        audio_dir=tmp_path,
        sample_idx=0,
        with_audio=False,
        audio_root="/data",
    )

    assert formatted["audio_filepath"] == "/data/numb3rs/Numb3rs/MONEY/MONEY_001.wav"
    assert formatted["audio_metadata"]["path"] == "/data/numb3rs/Numb3rs/MONEY/MONEY_001.wav"
    _assert_clean_audio_paths(formatted, "/data/numb3rs/")


def test_mmau_pro_audio_prefix(monkeypatch):
    """Cover the actual MMAU-Pro audio_path shape.

    The HF dataset records ``data/<uuid>.wav`` (the ``data/`` is the
    subdirectory in the dataset's zip layout), and after ``cp -r`` the
    file lives at ``<data_dir>/mmau-pro/data/<uuid>.wav``. The JSONL path
    must mirror that on-disk layout, so we keep the ``data/`` segment
    rather than stripping it.
    """
    _stub_audio_prepare_deps(monkeypatch)
    prepare = importlib.import_module("nemo_skills.dataset.mmau-pro.prepare")
    entry = {
        "answer": "A",
        "question": "What do you hear?",
        "choices": ["speech", "music"],
        "category": "closed_form",
        "audio_path": ["data/c93e3644-5227-4710-b27b-5c46750afbff.wav", "/already/absolute.wav"],
    }

    formatted = prepare.format_entry(entry, with_audio=True, audio_root="/data")

    assert formatted["audio_path"] == [
        "/data/mmau-pro/data/c93e3644-5227-4710-b27b-5c46750afbff.wav",
        "/data/mmau-pro/already/absolute.wav",
    ]
    assert formatted["messages"][0]["audios"][0]["path"] == (
        "/data/mmau-pro/data/c93e3644-5227-4710-b27b-5c46750afbff.wav"
    )


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
