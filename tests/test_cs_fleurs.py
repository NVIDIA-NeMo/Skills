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

"""Network-free unit tests for the CS-FLEURS benchmark's custom scoring logic.

These cover the pure helpers only (language/pair parsing, the CER-vs-WER
decision, and group-level aggregation). Actual data preparation requires a
HuggingFace download and is exercised separately by the GPU eval suite, where
``cs-fleurs`` is intentionally excluded from auto-prepare.
"""

from importlib import import_module

import pytest

languages = import_module("nemo_skills.dataset.cs-fleurs.languages")
audio_score = import_module("nemo_skills.dataset.cs-fleurs.audio_score")


class TestSplitPair:
    def test_hyphen_separator(self):
        assert languages.split_pair("ara-eng") == ("ara", "eng")

    def test_underscore_normalized_to_hyphen(self):
        assert languages.split_pair("cmn_eng") == ("cmn", "eng")

    def test_no_separator_is_matrix_only(self):
        assert languages.split_pair("eng") == ("eng", "")


class TestUsesCer:
    @pytest.mark.parametrize("matrix", ["cmn", "yue", "jpn", "kor", "tha", "lao", "mya", "khm", "vie"])
    def test_scriptio_continua_matrix_uses_cer(self, matrix):
        assert languages.uses_cer(matrix) is True

    @pytest.mark.parametrize("matrix", ["eng", "ara", "deu", "fra", "spa"])
    def test_space_delimited_matrix_uses_wer(self, matrix):
        assert languages.uses_cer(matrix) is False


class TestComputeScore:
    def test_empty_when_no_known_subsets(self):
        assert audio_score.compute_score({"some-other-bench": {"greedy": {"num_entries": 5}}}) == {}

    def test_entry_weighted_overall(self):
        combined = {
            "cs-fleurs.read": {"greedy": {"num_entries": 10, "wer": 20.0, "substitutions": 2}},
            "cs-fleurs.mms": {"greedy": {"num_entries": 30, "wer": 40.0, "substitutions": 6}},
        }
        result = audio_score.compute_score(combined)

        assert set(result["greedy"]) == {"read", "mms", "overall"}
        assert result["greedy"]["read"]["wer"] == 20.0
        assert result["greedy"]["mms"]["wer"] == 40.0

        overall = result["greedy"]["overall"]
        # entry-weighted: (20*10 + 40*30) / 40 == 35.0
        assert overall["wer"] == 35.0
        assert overall["num_entries"] == 40
        # summed (not weighted) error counts
        assert overall["substitutions"] == 8

    def test_zero_entry_subset_ignored(self):
        combined = {
            "cs-fleurs.read": {"greedy": {"num_entries": 0, "wer": 99.0}},
            "cs-fleurs.mms": {"greedy": {"num_entries": 5, "wer": 10.0}},
        }
        result = audio_score.compute_score(combined)
        assert "read" not in result["greedy"]
        assert result["greedy"]["overall"]["wer"] == 10.0
        assert result["greedy"]["overall"]["num_entries"] == 5
