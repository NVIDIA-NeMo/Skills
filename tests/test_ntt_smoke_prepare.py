# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import soundfile as sf


def _load_prepare_module():
    module_path = Path(__file__).parents[1] / "nemo_skills" / "dataset" / "ntt-smoke" / "prepare.py"
    spec = importlib.util.spec_from_file_location("ntt_smoke_prepare", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_eval_module():
    latex = types.ModuleType("latex2sympy2_extended")
    latex.NormalizationConfig = object
    latex.normalize_latex = lambda text, *args, **kwargs: text
    sys.modules.setdefault("latex2sympy2_extended", latex)

    math_verify = types.ModuleType("math_verify")
    math_verify.LatexExtractionConfig = object
    math_verify.StringExtractionConfig = object
    math_verify.parse = lambda *args, **kwargs: []
    math_verify.verify = lambda *args, **kwargs: False
    sys.modules.setdefault("math_verify", math_verify)

    contractions = types.ModuleType("contractions")
    contractions.fix = lambda text, *args, **kwargs: text
    sys.modules.setdefault("contractions", contractions)

    return __import__("importlib").import_module("nemo_skills.dataset.ntt-smoke.ntt_smoke_eval")


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row) + "\n")


def _write_audio(path: Path, seconds: float = 0.2, sr: int = 16000):
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.linspace(-0.1, 0.1, int(seconds * sr), dtype=np.float32)
    sf.write(path, samples, sr)


def _asr_row(dataset: str, sample_id: str, duration: float = 0.2):
    return {
        "task_type": "ASR",
        "expected_answer": f"transcript for {sample_id}",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. /no_think"},
            {
                "role": "user",
                "content": "Transcribe the following audio.",
                "audio": {
                    "path": f"/data/asr-leaderboard/data/{dataset}/{sample_id}.wav",
                    "duration": duration,
                },
            },
        ],
        "subset_for_metrics": dataset,
        "id": sample_id,
    }


def _context_row(mode: str):
    return {
        "messages": [
            {
                "role": "user",
                "content": f"{mode} prompt for Acme Rocket.",
                "audio": {"path": "/data/contextasr-bench/data/audio/context.wav", "duration": 0.2},
            }
        ],
        "expected_answer": "Acme Rocket launched.",
        "entity_list": ["Acme Rocket"],
        "domain_label": "Aerospace",
        "subset_for_metrics": "Aerospace",
        "uniq_id": "ctx-1",
        "duration": 0.2,
        "audio_filepath": "/data/contextasr-bench/data/audio/context.wav",
    }


def _make_source_root(root: Path):
    for dataset in [
        "librispeech_clean",
        "ami",
        "tedlium",
        "gigaspeech",
        "voxpopuli",
        "librispeech_other",
        "earnings22",
    ]:
        row = _asr_row(dataset, f"{dataset}-1")
        _write_jsonl(root / "asr-leaderboard" / f"{dataset}.jsonl", [row])
        _write_audio(root / "asr-leaderboard" / "data" / dataset / f"{dataset}-1.wav")

    musan_row = {
        "audio_path": ["/data/musan/noise/audio/noise.wav"],
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. /no_think"},
            {
                "role": "user",
                "content": "Transcribe the speech in this audio. If there is no speech, do not output anything.",
                "audio": {"path": "/data/musan/noise/audio/noise.wav", "duration": 0.2},
            },
        ],
        "expected_answer": "",
        "subset_for_metrics": "musan_noise",
        "sample_id": 1,
        "category": "noise",
        "task_type": "Hallucination",
        "audio_duration": 0.2,
    }
    _write_jsonl(root / "musan" / "test.jsonl", [musan_row])
    _write_audio(root / "musan" / "noise" / "audio" / "noise.wav")

    pc_row = {
        "audio_filepath": "/data/librispeech-pc/LibriSpeech/test-clean/pc.wav",
        "text": "Hello, World.",
        "expected_answer": "Hello, World.",
        "task_type": "ASR-PC",
        "sample_id": "pc",
        "split": "test-clean",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. /no_think"},
            {
                "role": "user",
                "content": "Transcribe the audio with proper punctuation and capitalization.",
                "audio": {"path": "/data/librispeech-pc/LibriSpeech/test-clean/pc.wav"},
            },
        ],
    }
    _write_jsonl(root / "librispeech-pc" / "test-clean.jsonl", [pc_row])
    _write_audio(root / "librispeech-pc" / "LibriSpeech" / "test-clean" / "pc.wav")

    for mode in ["contextless", "coarse", "fine"]:
        _write_jsonl(root / "contextasr-bench" / mode / "test.jsonl", [_context_row(mode)])
    _write_audio(root / "contextasr-bench" / "data" / "audio" / "context.wav")

    _write_jsonl(
        root / "gpqa" / "diamond.jsonl",
        [
            {
                "expected_answer": "A",
                "problem": "Which option is correct?\n\nA) yes\nB) no",
                "subset_for_metrics": "test",
            }
        ],
    )


def test_ntt_smoke_prepare_english_manifest(tmp_path):
    prepare = _load_prepare_module()
    source_root = tmp_path / "source"
    output_root = tmp_path / "out"
    _make_source_root(source_root)

    args = [
        "--source-data-dir",
        str(source_root),
        "--output-dir",
        str(output_root),
        "--audio-samples",
        "1",
        "--text-samples",
        "1",
        "--long-samples",
        "0",
        "--skip-multi",
    ]

    old_argv = __import__("sys").argv
    try:
        __import__("sys").argv = ["prepare.py", *args]
        prepare.main()
    finally:
        __import__("sys").argv = old_argv

    manifest = output_root / "en" / "test.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    subtasks = {row["ntt_subtask"] for row in rows}

    assert "asr.noisy_conversational" in subtasks
    assert "asr.noisy_media" in subtasks
    assert "hallucination.nonspeech" in subtasks
    assert "prompt_robustness" in subtasks
    assert "audio_instruction_following.punctuation_capitalization" in subtasks
    assert "context_biasing.fine" in subtasks
    assert "text.superficial" in subtasks
    assert all(row["origin_dataset"] for row in rows)
    assert (output_root / "data" / "noisy" / "en").exists()
    assert all(row.get("lang") for row in rows)


def test_ntt_smoke_prepare_prefers_preference_asr(tmp_path):
    prepare = _load_prepare_module()
    source_root = tmp_path / "source"
    preference_root = tmp_path / "preference-asr-bench"
    output_root = tmp_path / "out"
    _make_source_root(source_root)
    _write_jsonl(
        preference_root / "dev" / "pref.jsonl",
        [
            {
                "audio_filepath": "/tmp/pref.wav",
                "preference_text": "Use Preference Punctuation.",
                "text": "Use preference punctuation.",
                "instruction": "Use title case.",
                "preference_type": "case",
                "sub_category": ["proper_punctuated_capitalized_format"],
                "messages": [{"role": "user", "content": "Transcribe with punctuation.", "audio": {"path": "/tmp/pref.wav"}}],
            }
        ],
    )

    old_argv = __import__("sys").argv
    try:
        __import__("sys").argv = [
            "prepare.py",
            "--source-data-dir",
            str(source_root),
            "--preference-asr-dir",
            str(preference_root),
            "--output-dir",
            str(output_root),
            "--audio-samples",
            "1",
            "--preference-asr-groups",
            "case",
            "--preference-asr-prompt-variants",
            "original,direct",
            "--text-samples",
            "1",
            "--long-samples",
            "0",
            "--skip-multi",
        ]
        prepare.main()
    finally:
        __import__("sys").argv = old_argv

    rows = [json.loads(line) for line in (output_root / "en" / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    preference_rows = [row for row in rows if row["origin_dataset"] == "preference-asr-bench"]
    assert len(preference_rows) == 2
    assert {row["prompt_variant"] for row in preference_rows} == {
        "preference_asr.original",
        "preference_asr.direct",
    }
    assert preference_rows[0]["ntt_subtask"] == "audio_instruction_following.preference_asr.case"
    assert preference_rows[0]["expected_answer"] == "Use Preference Punctuation."
    assert preference_rows[0]["task_type"] == "PreferenceASR"


def test_ntt_smoke_prepare_spreads_long_rows():
    prepare = _load_prepare_module()
    rows = [{"ntt_subtask": "regular", "idx": idx} for idx in range(12)]
    rows.extend({"ntt_subtask": "asr.long", "idx": idx} for idx in range(3))

    balanced = prepare._balance_manifest_order(rows)

    long_positions = [idx for idx, row in enumerate(balanced) if row["ntt_subtask"] == "asr.long"]
    assert len(long_positions) == 3
    assert long_positions[0] > 0
    assert long_positions[-1] < len(balanced) - 1
    assert min(b - a for a, b in zip(long_positions, long_positions[1:])) > 1


def test_ntt_smoke_eval_normalizes_multilingual_language_metadata():
    evaluator = _load_eval_module()

    sample = evaluator._as_multilingual_asr_sample({"task_type": "ASR", "language": "cmn_hans_cn"})

    assert sample["task_type"] == "Multilingual-ASR"
    assert sample["extra_fields"]["src_lang"] == "zh"


def test_ntt_smoke_context_entity_scoring_ignores_unmatched_reference_entities():
    evaluator = _load_eval_module()
    sample = {
        "expected_answer": "The FDA reviewed a GDP contractionary forecast.",
        "entity_list": ["FDA", "GDP contraction"],
    }

    result = evaluator._evaluate_contextasr_sample(sample, "FDA approved the GDP contraction forecast.")

    assert result["ne_fnr_total"] == 1
    assert result["ne_fnr_hits"] == 1
    assert result["wer_correct_words"] >= 0


def test_ntt_smoke_metrics_track_correct_words():
    iso639 = types.ModuleType("iso639")
    iso639.__path__ = []
    iso639_exceptions = types.ModuleType("iso639.exceptions")
    iso639_exceptions.InvalidLanguageValue = ValueError
    iso639_iso639 = types.ModuleType("iso639.iso639")
    iso639_iso639.Lang = lambda value: types.SimpleNamespace(pt1=value)
    sys.modules.setdefault("iso639", iso639)
    sys.modules.setdefault("iso639.exceptions", iso639_exceptions)
    sys.modules.setdefault("iso639.iso639", iso639_iso639)

    langdetect = types.ModuleType("langdetect")
    langdetect.__path__ = []
    langdetect.DetectorFactory = types.SimpleNamespace(seed=0)
    langdetect.LangDetectException = ValueError
    langdetect.detect = lambda text: "en"
    langdetect_detector_factory = types.ModuleType("langdetect.detector_factory")
    langdetect_detector_factory.PROFILES_DIRECTORY = "/tmp"
    sys.modules.setdefault("langdetect", langdetect)
    sys.modules.setdefault("langdetect.detector_factory", langdetect_detector_factory)

    metrics_mod = __import__(
        "importlib"
    ).import_module("nemo_skills.dataset.ntt-smoke.ntt_smoke_metrics")
    metrics = metrics_mod.NTTSmokeMetrics()

    metrics.update(
        [
            {
                "generation": "hello there",
                "is_correct": True,
                "wer": 0.25,
                "wer_errors": 1,
                "wer_ref_words": 4,
                "wer_substitutions": 1,
                "wer_insertions": 0,
                "wer_deletions": 0,
            }
        ]
    )
    metrics.update(
        [
            {
                "generation": "goodbye now",
                "is_correct": False,
                "wer": 0.0,
                "wer_errors": 0,
                "wer_ref_words": 2,
                "wer_substitutions": 0,
                "wer_insertions": 0,
                "wer_deletions": 0,
            }
        ]
    )

    pass_metrics = metrics.get_metrics()["pass@1"]
    assert pass_metrics["correct_words"] == 5
    assert pass_metrics["substitutions"] == 1
    assert pass_metrics["insertions"] == 0
    assert pass_metrics["deletions"] == 0
    assert pass_metrics["wer_macro_ci95"] > 0
    assert pass_metrics["success_rate_ci95"] > 0
