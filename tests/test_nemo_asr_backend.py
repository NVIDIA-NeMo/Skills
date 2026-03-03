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

from recipes.multimodal.server.backends import GenerationRequest
from recipes.multimodal.server.backends.nemo_asr_backend import NeMoASRBackend, NeMoASRConfig


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
