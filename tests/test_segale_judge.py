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

"""Unit tests for the SEGALE judge script (segale_judge.py).

All SEGALE model calls are mocked so no GPU or model downloads are needed.

Tests cover:
  - Score phase correctly aggregates aligned spans to doc-level QE scores
  - Merge phase correctly writes scores back to original record order
  - .done marker is created after a successful merge
  - Merge returns early (preserving the embed checkpoint) when no scores exist
"""

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Fixture data: 5 sentences across 2 documents
#
# Mirrors the real document-level MT output format:
#   - seg_id=1 carries the full document translation in `generation`
#   - all other segments have generation="" (the model was given the full
#     document as one prompt and produced one generation, stored on the first
#     record; subsequent records preserve the source/reference structure)
# ---------------------------------------------------------------------------
DOC1_GENERATION = "The cat sat on the mat. She observed it carefully. Then she left it in peace."
DOC2_GENERATION = "It was a bright and sunny day. Children were playing in the park."

RECORDS = [
    # doc1 — 3 source/reference sentences
    {
        "source_language": "de_Latn",
        "target_language": "en_Latn",
        "text": "Die Katze saß auf der Matte. Sie beobachtete es aufmerksam. Dann ließ sie es in Ruhe.",
        "source_sentences": [
            "Die Katze saß auf der Matte.",
            "Sie beobachtete es aufmerksam.",
            "Dann ließ sie es in Ruhe.",
        ],
        "reference_sentences": [
            "The cat sat on the mat.",
            "She watched it carefully.",
            "Then she left it alone.",
        ],
        "generation": DOC1_GENERATION,
        "doc_id": "doc1",
        "seg_id": 1,
    },
    # doc2 — 2 source/reference sentences
    {
        "source_language": "de_Latn",
        "target_language": "en_Latn",
        "text": "Es war ein heller Tag. Kinder spielten im Park.",
        "source_sentences": [
            "Es war ein heller Tag.",
            "Kinder spielten im Park.",
        ],
        "reference_sentences": [
            "It was a bright day.",
            "Children played in the park.",
        ],
        "generation": DOC2_GENERATION,
        "doc_id": "doc2",
        "seg_id": 1,
    },
]

# Pre-baked aligned spans that the score phase will process.
# 3 spans for doc1, 2 spans for doc2 (1:1 alignment for simplicity).
MOCK_ALIGNED_SPANS = [
    {
        "doc_id": "doc1",
        "sys_id": "nemo_skills",
        "src": RECORDS[0]["source_sentences"][0],
        "ref": RECORDS[0]["reference_sentences"][0],
        "tgt": "The cat sat on the mat.",
        "seg_id": 1,
    },
    {
        "doc_id": "doc1",
        "sys_id": "nemo_skills",
        "src": RECORDS[0]["source_sentences"][1],
        "ref": RECORDS[0]["reference_sentences"][1],
        "tgt": "She observed it carefully.",
        "seg_id": 2,
    },
    {
        "doc_id": "doc1",
        "sys_id": "nemo_skills",
        "src": RECORDS[0]["source_sentences"][2],
        "ref": RECORDS[0]["reference_sentences"][2],
        "tgt": "Then she left it in peace.",
        "seg_id": 3,
    },
    {
        "doc_id": "doc2",
        "sys_id": "nemo_skills",
        "src": RECORDS[1]["source_sentences"][0],
        "ref": RECORDS[1]["reference_sentences"][0],
        "tgt": "It was a bright and sunny day.",
        "seg_id": 1,
    },
    {
        "doc_id": "doc2",
        "sys_id": "nemo_skills",
        "src": RECORDS[1]["source_sentences"][1],
        "ref": RECORDS[1]["reference_sentences"][1],
        "tgt": "Children were playing in the park.",
        "seg_id": 2,
    },
]

# QE scores the mock returns (one per aligned span).
MOCK_COMET_QE_SCORES = [0.83, 0.86, 0.80, 0.87, 0.84]
MOCK_METRICX_QE_SCORES = [1.5, 1.4, 1.6, 1.1, 1.2]

# Expected doc-level averages.  doc1: spans 0-2 — doc2: spans 3-4
DOC1_COMET_QE = sum(MOCK_COMET_QE_SCORES[:3]) / 3
DOC1_METRICX_QE = sum(MOCK_METRICX_QE_SCORES[:3]) / 3
DOC2_COMET_QE = sum(MOCK_COMET_QE_SCORES[3:]) / 2
DOC2_METRICX_QE = sum(MOCK_METRICX_QE_SCORES[3:]) / 2


# ---------------------------------------------------------------------------
# Stubs for heavy optional dependencies
# ---------------------------------------------------------------------------


def _make_segale_align_stub():
    stub = types.ModuleType("segale_align")
    stub.VERBOSE = 0
    stub.SPACY = "ersatz"
    stub.STOP_JUMP = 0.15
    stub.COST_MIN = 0.30
    stub.COST_MAX = 0.30
    stub.init_config = MagicMock()
    stub.load_alternative_model = MagicMock(return_value=(None, MagicMock()))
    stub.merge_system_entries = MagicMock(return_value=[])
    stub.merge_ref_entries = MagicMock(return_value=[])
    stub.combine_system_ref = MagicMock(return_value=[])
    stub.prepare_doc_windows = MagicMock(return_value=None)
    return stub


def _make_laser_stub():
    stub = types.ModuleType("laser_encoders")
    stub.LaserEncoderPipeline = MagicMock(return_value=MagicMock())
    return stub


def _make_segale_eval_stub():
    """Stub for segale_eval; only QE scorers are used by the current pipeline."""
    stub = types.ModuleType("segale_eval")
    stub.run_comet_qe_evaluation = MagicMock(return_value=MOCK_COMET_QE_SCORES)
    stub.run_metricx_qe_evaluation = MagicMock(return_value=MOCK_METRICX_QE_SCORES)
    return stub


# ---------------------------------------------------------------------------
# Module loader
# ---------------------------------------------------------------------------


def _import_judge_with_stubs():
    """Import segale_judge with SEGALE library stubs injected into sys.modules."""
    sys.modules.pop("segale_judge", None)
    sys.modules["segale_align"] = _make_segale_align_stub()
    sys.modules["laser_encoders"] = _make_laser_stub()
    sys.modules["segale_eval"] = _make_segale_eval_stub()

    evaluator_dir = str(Path(__file__).parent.parent / "nemo_skills" / "evaluation" / "evaluator")
    if evaluator_dir not in sys.path:
        sys.path.insert(0, evaluator_dir)

    import segale_judge

    return segale_judge


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path, records):
    with open(path, "wt", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRunScorePhase:
    """_run_score_phase aggregates aligned spans to doc-level QE scores and writes per-lang output."""

    def setup_method(self):
        self.judge = _import_judge_with_stubs()

    def _setup(self, tmp_path):
        """Write input records and aligned spans; return (input_file, output_dir)."""
        input_file = tmp_path / "input.jsonl"
        _write_jsonl(input_file, RECORDS)

        align_dir = tmp_path / "segale_intermediate" / "align" / "en_Latn"
        align_dir.mkdir(parents=True)
        _write_jsonl(align_dir / "aligned_spans.jsonl", MOCK_ALIGNED_SPANS)
        (align_dir / "align.done").touch()

        return input_file, tmp_path

    def _read_per_lang(self, output_dir):
        return _read_jsonl(output_dir / "per_lang" / "en_Latn" / "output.jsonl")

    def test_comet_qe_average_doc1(self, tmp_path):
        input_file, output_dir = self._setup(tmp_path)
        self.judge._run_score_phase(input_file=input_file, output_dir=output_dir)

        doc1 = next(r for r in self._read_per_lang(output_dir) if r.get("doc_id") == "doc1")
        assert abs(doc1["segale_comet_qe"] - DOC1_COMET_QE) < 1e-9

    def test_comet_qe_average_doc2(self, tmp_path):
        input_file, output_dir = self._setup(tmp_path)
        self.judge._run_score_phase(input_file=input_file, output_dir=output_dir)

        doc2 = next(r for r in self._read_per_lang(output_dir) if r.get("doc_id") == "doc2")
        assert abs(doc2["segale_comet_qe"] - DOC2_COMET_QE) < 1e-9

    def test_all_metrics_present(self, tmp_path):
        input_file, output_dir = self._setup(tmp_path)
        self.judge._run_score_phase(input_file=input_file, output_dir=output_dir)

        for record in self._read_per_lang(output_dir):
            for field in ("segale_comet_qe", "segale_metricx_qe", "segale_lang_fidelity", "segale_total_seg"):
                assert field in record, f"{field} missing from scored record (doc_id={record.get('doc_id')})"

    def test_total_seg_count(self, tmp_path):
        """segale_total_seg counts all spans (including misaligned), not just valid ones."""
        input_file, output_dir = self._setup(tmp_path)
        self.judge._run_score_phase(input_file=input_file, output_dir=output_dir)

        out = self._read_per_lang(output_dir)
        doc1 = next(r for r in out if r.get("doc_id") == "doc1")
        doc2 = next(r for r in out if r.get("doc_id") == "doc2")
        assert doc1["segale_total_seg"] == 3  # 3 spans total for doc1
        assert doc2["segale_total_seg"] == 2  # 2 spans total for doc2

    def test_done_marker_created(self, tmp_path):
        input_file, output_dir = self._setup(tmp_path)
        self.judge._run_score_phase(input_file=input_file, output_dir=output_dir)

        assert (output_dir / "per_lang" / "en_Latn" / "output.jsonl.done").exists()


class TestMergePerLanguageOutputs:
    """merge_per_language_outputs combines per-lang scores into the final output file."""

    def setup_method(self):
        self.judge = _import_judge_with_stubs()

    def _scored_records(self):
        """Return RECORDS with doc-level scores pre-applied (as _run_score_phase would write)."""
        return [
            {
                **RECORDS[0],
                "segale_comet_qe": DOC1_COMET_QE,
                "segale_metricx_qe": DOC1_METRICX_QE,
                "segale_lang_fidelity": 1.0,
                "segale_total_seg": 3,
                "segale_misaligned_seg": 0,  # all mock spans have positive scores
            },
            {
                **RECORDS[1],
                "segale_comet_qe": DOC2_COMET_QE,
                "segale_metricx_qe": DOC2_METRICX_QE,
                "segale_lang_fidelity": 1.0,
                "segale_total_seg": 2,
                "segale_misaligned_seg": 0,  # all mock spans have positive scores
            },
        ]

    def test_scores_merged_into_output(self, tmp_path):
        input_file = tmp_path / "input.jsonl"
        langs_dir = tmp_path / "per_lang"
        output_file = tmp_path / "output.jsonl"

        _write_jsonl(input_file, RECORDS)
        lang_dir = langs_dir / "en_Latn"
        lang_dir.mkdir(parents=True)
        _write_jsonl(lang_dir / "output.jsonl", self._scored_records())

        self.judge.merge_per_language_outputs(input_file, langs_dir, output_file)

        out = _read_jsonl(output_file)
        doc1 = next(r for r in out if r.get("doc_id") == "doc1")
        doc2 = next(r for r in out if r.get("doc_id") == "doc2")
        assert abs(doc1["segale_comet_qe"] - DOC1_COMET_QE) < 1e-9
        assert abs(doc2["segale_comet_qe"] - DOC2_COMET_QE) < 1e-9

    def test_done_marker_created(self, tmp_path):
        input_file = tmp_path / "input.jsonl"
        langs_dir = tmp_path / "per_lang"
        output_file = tmp_path / "output.jsonl"

        _write_jsonl(input_file, RECORDS)
        lang_dir = langs_dir / "en_Latn"
        lang_dir.mkdir(parents=True)
        _write_jsonl(lang_dir / "output.jsonl", self._scored_records())

        self.judge.merge_per_language_outputs(input_file, langs_dir, output_file)

        assert Path(str(output_file) + ".done").exists()

    def test_empty_score_map_returns_early(self, tmp_path):
        """When no per-lang scores exist, merge returns early to preserve the embed checkpoint."""
        input_file = tmp_path / "input.jsonl"
        langs_dir = tmp_path / "per_lang"
        output_file = tmp_path / "output.jsonl"

        _write_jsonl(input_file, RECORDS)
        langs_dir.mkdir(parents=True)
        # No per-lang output.jsonl files — simulates upstream timeout/failure

        self.judge.merge_per_language_outputs(input_file, langs_dir, output_file)

        assert not output_file.exists(), "output.jsonl must not be written when score_map is empty"
        assert not Path(str(output_file) + ".done").exists(), ".done must not be written when score_map is empty"
