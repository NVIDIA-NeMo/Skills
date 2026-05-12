# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0

import importlib
import asyncio
import inspect
import json
import subprocess
import types
import wave
from pathlib import Path

import pytest

from nemo_skills.dataset.utils import get_dataset_module
from nemo_skills.evaluation.evaluator.librispeechmix_sot import LibriSpeechMixSOTEvaluator
from nemo_skills.evaluation.metrics.librispeechmix_sot_metrics import LibriSpeechMixSOTMetrics
from nemo_skills.evaluation.metrics.librispeechmix_sot_utils import cpwer, parse_sot_speaker_streams


prepare = importlib.import_module("nemo_skills.dataset.librispeechmix-sot.prepare")


def _write_wav(path: Path, duration: float = 0.05, sample_rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(duration * sample_rate)
    with wave.open(str(path), "wb") as wavf:
        wavf.setnchannels(1)
        wavf.setsampwidth(2)
        wavf.setframerate(sample_rate)
        wavf.writeframes(b"\x00\x00" * frames)


def _row(duration_a=5.0, delay_b=2.0, duration_b=3.0):
    return {
        "id": "test-clean-2mix/test-clean-2mix-0000",
        "mixed_wav": "test-clean-2mix/test-clean-2mix-0000.wav",
        "texts": ["HELLO, WORLD", "GOOD MORNING"],
        "wavs": ["test-clean/111/1/111-1-0000.wav", "test-clean/222/2/222-2-0000.wav"],
        "delays": [0.0, delay_b],
        "speakers": ["111", "222"],
        "durations": [duration_a, duration_b],
        "genders": ["m", "f"],
        "speaker_profile": [],
        "speaker_profile_index": [0, 1],
    }


def test_group_registration():
    group_module, _ = get_dataset_module("librispeechmix-sot")
    assert group_module.IS_BENCHMARK_GROUP is True
    assert "librispeechmix-sot.under20s-test-clean-3mix" in group_module.BENCHMARKS

    sub_module, _ = get_dataset_module("librispeechmix-sot.under20s-test-clean-1mix")
    assert sub_module.METRICS_TYPE == "librispeechmix_sot"
    assert "++eval_type=librispeechmix_sot" in sub_module.EVAL_ARGS


def test_under_over_split_threshold():
    assert prepare.duration_bucket(20.0) == "under20s"
    assert prepare.duration_bucket(20.0001) == "over20s"


def test_rttm_generation_and_stable_sot_tags():
    row = {
        **_row(),
        "texts": ["SECOND", "FIRST"],
        "speakers": ["b_spk", "a_spk"],
        "delays": [1.0, 0.0],
        "durations": [2.0, 3.0],
    }

    assert prepare.speaker_tag_map(row) == {"a_spk": "s0", "b_spk": "s1"}
    assert prepare.sot_text(row) == "[s0] first [s1] second"
    assert prepare.num_speaker_changes(row) == 1

    lines = prepare.rttm_lines(row, "mix0")
    assert lines == [
        "SPEAKER mix0 1 1.000 2.000 <NA> <NA> b_spk <NA> <NA>\n",
        "SPEAKER mix0 1 0.000 3.000 <NA> <NA> a_spk <NA> <NA>\n",
    ]


def test_prepare_writes_under_and_over_manifests(tmp_path):
    data_dir = tmp_path / "data"
    rows = [_row(duration_a=5.0, delay_b=1.0, duration_b=3.0), _row(duration_a=21.0, delay_b=0.0, duration_b=1.0)]
    counts = prepare.prepare_split_mix(
        split="test-clean",
        mix="2mix",
        rows=rows,
        output_dir=tmp_path / "benchmark",
        data_dir=data_dir,
        audio_prefix=data_dir / "audio",
        no_audio=True,
        max_samples=1,
        librispeech_root=data_dir / "LibriSpeech",
    )

    assert counts == {"under20s-test-clean-2mix": 1, "over20s-test-clean-2mix": 1}
    under_record = json.loads((tmp_path / "benchmark/under20s-test-clean-2mix/test.jsonl").read_text().splitlines()[0])
    over_record = json.loads((tmp_path / "benchmark/over20s-test-clean-2mix/test.jsonl").read_text().splitlines()[0])

    assert under_record["duration"] <= 20.0
    assert over_record["duration"] > 20.0
    assert under_record["source_lang"] == "en"
    assert under_record["taskname"] == "asr"
    assert under_record["pnc"] == "no"
    assert Path(under_record["rttm_filepath"]).exists()
    assert Path(under_record["reference_seglst_filepath"]).exists()
    assert under_record["messages"][0]["audio"]["path"] == under_record["audio_filepath"]


def test_materialize_mixed_wav(tmp_path):
    data_dir = tmp_path / "data"
    libri_root = data_dir / "LibriSpeech"
    _write_wav(libri_root / "test-clean/111/1/111-1-0000.wav")
    _write_wav(libri_root / "test-clean/222/2/222-2-0000.wav")

    record = prepare.build_record(
        _row(duration_a=0.05, delay_b=0.01, duration_b=0.05),
        split="test-clean",
        mix="2mix",
        data_dir=data_dir,
        audio_prefix=data_dir / "audio",
        no_audio=False,
        librispeech_root=libri_root,
    )

    assert Path(record["audio_filepath"]).exists()
    assert record["num_speakers"] == 2


def test_sot_parser_and_cpwer_permutation_invariance():
    parsed = parse_sot_speaker_streams("[s0] Hello, [s1] YES! [s0] again")
    assert parsed == {"s0": "hello again", "s1": "yes"}

    result = cpwer("[s0] hello world [s1] good morning", "[s0] good morning [s1] hello world")
    assert result["errors"] == 0
    assert result["cpwer"] == 0.0


@pytest.mark.parametrize(
    ("hypothesis", "errors", "ref_words"),
    [
        ("[s0] hello", 2, 3),
        ("[s0] hello [s1] there [s2] extra words", 3, 3),
        ("", 3, 3),
    ],
)
def test_cpwer_missing_extra_empty_hypothesis(hypothesis, errors, ref_words):
    result = cpwer("[s0] hello [s1] there now", hypothesis)
    assert result["errors"] == errors
    assert result["ref_words"] == ref_words


def test_cpwer_punctuation_case_normalization():
    result = cpwer("[s0] Hello, WORLD!", "[s0] hello world")
    assert result["errors"] == 0


def test_metrics_use_corpus_level_raw_counts():
    metrics = LibriSpeechMixSOTMetrics(compute_no_answer=False, max_k=1)
    metrics.update(
        [
            {
                "generation": "wrong",
                "is_correct": False,
                "cpwer_errors": 1,
                "cpwer_substitutions": 1,
                "cpwer_insertions": 0,
                "cpwer_deletions": 0,
                "cpwer_ref_words": 1,
            }
        ]
    )
    metrics.update(
        [
            {
                "generation": "correct",
                "is_correct": True,
                "cpwer_errors": 0,
                "cpwer_substitutions": 0,
                "cpwer_insertions": 0,
                "cpwer_deletions": 0,
                "cpwer_ref_words": 100,
            }
        ]
    )

    output = metrics.get_metrics()["pass@1"]
    assert output["cpwer_errors"] == 1
    assert output["cpwer_ref_words"] == 101
    assert output["cpwer"] == 0.99


def test_evaluator_reads_pred_text_before_generation():
    evaluator = LibriSpeechMixSOTEvaluator({})
    result = asyncio.run(
        evaluator.eval_single({"text": "[s0] hello", "pred_text": "[s0] hello", "generation": "[s0] wrong"})
    )

    assert result["cpwer_errors"] == 0
    assert result["is_correct"] is True


def test_rtmtasr_command_and_generation_fake(monkeypatch, tmp_path):
    module = importlib.import_module("nemo_skills.inference.librispeechmix_rtmtasr")
    output_file = tmp_path / "out.jsonl"
    cfg = module.RTMTASRGenerationConfig(
        input_file="input.jsonl",
        output_file=str(output_file),
        model_path="model.nemo",
        nemo_root="/repo/NeMo",
    )

    command = module.build_rtmtasr_command(cfg)
    assert "/repo/NeMo/examples/asr/transcribe_speech_rtmtasr.py" in command[2]
    assert "spk_supervision=rttm" in command
    assert "model_path=model.nemo" in command
    assert "+prompt.pnc=no" in command

    def fake_run(cmd, check, env):
        assert cmd == command
        output_file.write_text(json.dumps({"text": "[s0] hi", "pred_text": "[s0] hi"}) + "\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module.RTMTASRGenerationTask(cfg).generate()
    row = json.loads(output_file.read_text())
    assert row["generation"] == "[s0] hi"


def test_multitalker_parakeet_command_and_generation_fake(monkeypatch, tmp_path):
    module = importlib.import_module("nemo_skills.inference.librispeechmix_multitalker_parakeet")
    input_file = tmp_path / "input.jsonl"
    output_file = tmp_path / "out.jsonl"
    input_file.write_text(
        json.dumps({"sample_id": "mix0", "audio_filepath": "/audio/mix0.wav", "text": "[s0] hi [s1] there"}) + "\n"
    )
    cfg = module.MultitalkerParakeetGenerationConfig(
        input_file=str(input_file),
        output_file=str(output_file),
        nemo_root="/repo/NeMo",
        asr_model="asr.nemo",
        diar_model="diar.nemo",
    )

    command = module.build_multitalker_parakeet_command(cfg)
    assert "/repo/NeMo/examples/asr/asr_cache_aware_streaming/speech_to_text_multitalker_streaming_infer.py" in command[1]
    assert "asr_model=asr.nemo" in command
    assert "diar_model=diar.nemo" in command

    def fake_run(cmd, check, env):
        assert cmd == command
        output_file.write_text(
            json.dumps(
                [
                    {"session_id": "mix0", "speaker": "speaker_a", "words": "hi", "start_time": 0.0, "end_time": 0.5},
                    {
                        "session_id": "mix0",
                        "speaker": "speaker_b",
                        "words": "there",
                        "start_time": 0.5,
                        "end_time": 1.0,
                    },
                ]
            )
            + "\n"
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    module.MultitalkerParakeetGenerationTask(cfg).generate()
    row = json.loads(output_file.read_text())
    assert row["text"] == "[s0] hi [s1] there"
    assert row["generation"] == "[s0] hi [s1] there"
    assert row["hypothesis_seglst"][0]["speaker"] == "speaker_a"


def test_multitalker_parakeet_unified_backend_fake(monkeypatch, tmp_path):
    backends = importlib.import_module("recipes.multimodal.server.backends")
    module = importlib.import_module("recipes.multimodal.server.backends.multitalker_parakeet_backend")
    assert backends.get_backend("multitalker_parakeet").__name__ == "MultitalkerParakeetBackend"
    assert not hasattr(module, "subprocess")
    backend_source = inspect.getsource(module.MultitalkerParakeetBackend.generate)
    backend_source += inspect.getsource(module.MultitalkerParakeetBackend._run_inprocess_manifest)
    assert "subprocess.run" not in backend_source

    audio_path = tmp_path / "input.wav"
    _write_wav(audio_path, duration=0.05)

    cfg = module.MultitalkerParakeetConfig(
        model_path="asr.nemo",
        diar_model="diar.nemo",
        nemo_root=str(tmp_path),
        resolve_hf_models=False,
        batch_size=2,
    )
    backend = module.MultitalkerParakeetBackend(cfg)
    backend._is_loaded = True

    def fake_run_inprocess(manifest_file, batch_size):
        assert batch_size == 1
        manifest_rows = [json.loads(line) for line in Path(manifest_file).read_text().splitlines()]
        assert len(manifest_rows) == 1
        assert Path(manifest_rows[0]["audio_filepath"]).exists()
        assert manifest_rows[0]["duration"] > 0
        return [
            {"session_id": "req_0000", "speaker": "speaker_0", "words": "hi", "start_time": 0.0, "end_time": 0.5},
            {
                "session_id": "req_0000",
                "speaker": "speaker_1",
                "words": "there",
                "start_time": 0.5,
                "end_time": 1.0,
            },
        ]

    monkeypatch.setattr(backend, "_run_inprocess_manifest", fake_run_inprocess)
    result = backend.generate([module.GenerationRequest(audio_bytes=audio_path.read_bytes())])[0]
    assert result.error is None
    assert result.text == "[s0] hi [s1] there"
    assert result.debug_info["backend"] == "multitalker_parakeet"
    assert result.debug_info["persistent_inprocess"] is True


def test_multitalker_parakeet_backend_coerces_mutating_seglst_return():
    module = importlib.import_module("recipes.multimodal.server.backends.multitalker_parakeet_backend")
    entries = [{"session_id": "req_0000", "speaker": "speaker_0", "words": "hi"}]
    streamer = types.SimpleNamespace(instance_manager=types.SimpleNamespace(seglst_dict_list=entries))
    assert module.MultitalkerParakeetBackend._coerce_batch_entries(streamer, None) == entries
