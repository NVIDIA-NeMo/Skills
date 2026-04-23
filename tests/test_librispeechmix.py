# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import json
from pathlib import Path

import pytest
import soundfile as sf

from nemo_skills.dataset.utils import get_dataset_module
from nemo_skills.evaluation.evaluator.audio import (
    evaluate_librispeechmix_asr,
    evaluate_librispeechmix_sa_asr,
)


def _write_tone(path: Path, duration_sec: float = 0.1, sample_rate: int = 16000) -> None:
    import numpy as np

    num_samples = int(duration_sec * sample_rate)
    audio = np.sin(np.linspace(0, 2 * np.pi, num_samples, endpoint=False)).astype("float32")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate)


def test_librispeechmix_group_registration():
    group_module, _ = get_dataset_module("librispeechmix")
    assert group_module.IS_BENCHMARK_GROUP is True
    assert len(group_module.BENCHMARKS) == 12

    sub_module, _ = get_dataset_module("librispeechmix.sa-asr-test-clean-3mix")
    assert sub_module.METRICS_TYPE == "audio"
    assert "++eval_type=audio" in sub_module.EVAL_ARGS


def test_librispeechmix_asr_uses_permutation_invariant_wer():
    metrics = evaluate_librispeechmix_asr(
        references=["hello world", "good morning"],
        hypothesis="good morning\nhello world",
        normalization_mode="none",
    )

    assert metrics["wer"] == pytest.approx(0.0)
    assert metrics["wer_errors"] == 0
    assert metrics["wer_ref_words"] == 4
    assert metrics["is_correct"] is True


def test_librispeechmix_sa_asr_uses_speaker_indices_not_order():
    metrics = evaluate_librispeechmix_sa_asr(
        references=["hello world", "good morning"],
        speaker_profile_index=[5, 2],
        hypothesis="speaker_2: good morning\nspeaker_5: hello world",
        normalization_mode="none",
    )

    assert metrics["wer"] == pytest.approx(0.0)
    assert metrics["wer_errors"] == 0
    assert metrics["wer_ref_words"] == 4
    assert metrics["is_correct"] is True


def test_librispeechmix_prepare_records_writes_absolute_paths(tmp_path):
    from nemo_skills.dataset.librispeechmix.prepare import (
        build_output_record,
        write_benchmark_records,
    )

    data_root = tmp_path / "data"
    raw_root = data_root / "raw" / "LibriSpeech"
    audio_root = data_root / "audio"
    output_root = data_root / "librispeechmix"

    _write_tone(raw_root / "dev-clean" / "111" / "1" / "111-1-0000.flac")
    _write_tone(raw_root / "dev-clean" / "222" / "2" / "222-2-0000.flac")
    _write_tone(raw_root / "dev-clean" / "333" / "3" / "333-3-0000.flac")
    _write_tone(raw_root / "dev-clean" / "444" / "4" / "444-4-0000.flac")

    row = {
        "id": "dev-clean-2mix/dev-clean-2mix-0000",
        "mixed_wav": "dev-clean-2mix/dev-clean-2mix-0000.wav",
        "texts": ["HELLO WORLD", "GOOD MORNING"],
        "speaker_profile": [
            ["dev-clean/111/1/111-1-0000.wav", "dev-clean/111/1/111-1-0000.wav"],
            ["dev-clean/222/2/222-2-0000.wav", "dev-clean/222/2/222-2-0000.wav"],
            ["dev-clean/333/3/333-3-0000.wav", "dev-clean/333/3/333-3-0000.wav"],
            ["dev-clean/444/4/444-4-0000.wav", "dev-clean/444/4/444-4-0000.wav"],
            ["dev-clean/111/1/111-1-0000.wav", "dev-clean/111/1/111-1-0000.wav"],
            ["dev-clean/222/2/222-2-0000.wav", "dev-clean/222/2/222-2-0000.wav"],
            ["dev-clean/333/3/333-3-0000.wav", "dev-clean/333/3/333-3-0000.wav"],
            ["dev-clean/444/4/444-4-0000.wav", "dev-clean/444/4/444-4-0000.wav"],
        ],
        "speaker_profile_index": [1, 3],
        "wavs": ["dev-clean/111/1/111-1-0000.wav", "dev-clean/222/2/222-2-0000.wav"],
        "delays": [0.0, 0.02],
        "speakers": ["111", "222"],
        "durations": [0.1, 0.1],
        "genders": ["m", "f"],
    }

    record = build_output_record(
        row=row,
        mode="sa-asr",
        mixed_audio_path=audio_root / "mixed" / "dev-clean-2mix" / "dev-clean-2mix-0000.wav",
        mixed_duration=0.12,
        audio_prefix=audio_root.resolve(),
        source_audio_root=audio_root / "source",
        raw_librispeech_root=raw_root,
        materialize_audio=True,
    )

    output_file = output_root / "sa-asr-dev-clean-2mix" / "test.jsonl"
    write_benchmark_records(output_file, [record])

    with open(output_file, encoding="utf-8") as fin:
        saved = json.loads(fin.readline())

    assert Path(saved["messages"][-1]["audio"]["path"]).is_absolute()
    assert Path(saved["messages"][1]["audios"][0]["path"]).is_absolute()
    assert (audio_root / "mixed" / "dev-clean-2mix" / "dev-clean-2mix-0000.wav").exists()
