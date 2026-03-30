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

"""
TDD test for SEGALE metric integration in TranslationMetrics.

tests than 'ns summarize_results' works as expected once the sentences
have been properly segmented and scored on a sentence level.
"""

import json

from nemo_skills.pipeline.summarize_results import summarize_results

# Pre-computed SEGALE scores that the judge script would write into output.jsonl.
# Two language pairs: de_Latn->en_Latn and zho_Hans->en_Latn.
#
# Format mirrors SEGALE's native input: one record per sentence.
#   source      — one source sentence
#   translation — one reference sentence
#   generation  — one sentence of the model's MT output
#   doc_id      — groups sentences into documents
#   seg_id      — sentence index within the document
#   sys_id      — system/model identifier
#   segale_*    — document-level score written by the judge script to every
#                 sentence record in the document (same value per doc).
#                 Aggregation weighting across doc sizes is a TODO.
PREDICTIONS = [
    # --- de_Latn -> en_Latn, doc 1 (3 sentences) ---
    {
        "source_language": "de_Latn",
        "target_language": "en_Latn",
        "source": "Die Katze saß auf der Matte.",
        "translation": "The cat sat on the mat.",
        "generation": "The cat sat on the mat.",
        "doc_id": "de_doc1",
        "seg_id": 0,
        "sys_id": "test_model",
        "segale_comet": 0.88,
        "segale_comet_qe": 0.83,
        "segale_metricx": 1.2,
        "segale_metricx_qe": 1.5,
    },
    {
        "source_language": "de_Latn",
        "target_language": "en_Latn",
        "source": "Sie beobachtete es aufmerksam von der anderen Seite des Zimmers.",
        "translation": "She watched it carefully from across the room.",
        "generation": "She watched it carefully from across the room.",
        "doc_id": "de_doc1",
        "seg_id": 1,
        "sys_id": "test_model",
        "segale_comet": 0.88,
        "segale_comet_qe": 0.83,
        "segale_metricx": 1.2,
        "segale_metricx_qe": 1.5,
    },
    {
        "source_language": "de_Latn",
        "target_language": "en_Latn",
        "source": "Nach einer Weile beschloss sie, es in Ruhe zu lassen.",
        "translation": "After some time, she decided to leave it in peace.",
        "generation": "After a while, she decided to leave it alone.",
        "doc_id": "de_doc1",
        "seg_id": 2,
        "sys_id": "test_model",
        "segale_comet": 0.88,
        "segale_comet_qe": 0.83,
        "segale_metricx": 1.2,
        "segale_metricx_qe": 1.5,
    },
    # --- de_Latn -> en_Latn, doc 2 (2 sentences) ---
    {
        "source_language": "de_Latn",
        "target_language": "en_Latn",
        "source": "Es war ein heller und sonniger Tag.",
        "translation": "It was a bright sunny day.",
        "generation": "It was a bright and sunny day.",
        "doc_id": "de_doc2",
        "seg_id": 0,
        "sys_id": "test_model",
        "segale_comet": 0.90,
        "segale_comet_qe": 0.87,
        "segale_metricx": 0.9,
        "segale_metricx_qe": 1.1,
    },
    {
        "source_language": "de_Latn",
        "target_language": "en_Latn",
        "source": "Kinder spielten im Park, während ihre Eltern auf den Bänken saßen.",
        "translation": "Children played in the park while their parents rested on the benches.",
        "generation": "Children played in the park while their parents sat on the benches.",
        "doc_id": "de_doc2",
        "seg_id": 1,
        "sys_id": "test_model",
        "segale_comet": 0.90,
        "segale_comet_qe": 0.87,
        "segale_metricx": 0.9,
        "segale_metricx_qe": 1.1,
    },
    # --- zho_Hans -> en_Latn, doc 1 (3 sentences) ---
    {
        "source_language": "zho_Hans",
        "target_language": "en_Latn",
        "source": "去年经济大幅增长。",
        "translation": "The economy saw significant growth last year.",
        "generation": "The economy grew significantly last year.",
        "doc_id": "zh_doc1",
        "seg_id": 0,
        "sys_id": "test_model",
        "segale_comet": 0.82,
        "segale_comet_qe": 0.78,
        "segale_metricx": 2.3,
        "segale_metricx_qe": 2.7,
    },
    {
        "source_language": "zho_Hans",
        "target_language": "en_Latn",
        "source": "第三季度出口达到创纪录的高位。",
        "translation": "Exports hit a record high in the third quarter.",
        "generation": "Exports reached a record high in the third quarter.",
        "doc_id": "zh_doc1",
        "seg_id": 1,
        "sys_id": "test_model",
        "segale_comet": 0.82,
        "segale_comet_qe": 0.78,
        "segale_metricx": 2.3,
        "segale_metricx_qe": 2.7,
    },
    {
        "source_language": "zho_Hans",
        "target_language": "en_Latn",
        "source": "分析师预计这一趋势将延续到明年。",
        "translation": "Analysts expect this trend to continue into next year.",
        "generation": "Analysts expect the trend to continue into next year.",
        "doc_id": "zh_doc1",
        "seg_id": 2,
        "sys_id": "test_model",
        "segale_comet": 0.82,
        "segale_comet_qe": 0.78,
        "segale_metricx": 2.3,
        "segale_metricx_qe": 2.7,
    },
    # --- zho_Hans -> en_Latn, doc 2 (2 sentences) ---
    {
        "source_language": "zho_Hans",
        "target_language": "en_Latn",
        "source": "科学家在雨林中发现了一个新物种。",
        "translation": "Scientists discovered a new species in the rainforest.",
        "generation": "Scientists have discovered a new species in the rainforest.",
        "doc_id": "zh_doc2",
        "seg_id": 0,
        "sys_id": "test_model",
        "segale_comet": 0.86,
        "segale_comet_qe": 0.83,
        "segale_metricx": 1.4,
        "segale_metricx_qe": 1.6,
    },
    {
        "source_language": "zho_Hans",
        "target_language": "en_Latn",
        "source": "介绍这些发现的会议将于三月举行。",
        "translation": "The conference presenting these findings is scheduled for March.",
        "generation": "The conference to present the findings will be held in March.",
        "doc_id": "zh_doc2",
        "seg_id": 1,
        "sys_id": "test_model",
        "segale_comet": 0.86,
        "segale_comet_qe": 0.83,
        "segale_metricx": 1.4,
        "segale_metricx_qe": 1.6,
    },
]

SEGALE_FIELDS = ["segale_comet", "segale_comet_qe", "segale_metricx", "segale_metricx_qe"]


def test_segale_fields_in_summarize_results(tmp_path):
    """SEGALE scores pre-written by the judge in output.jsonl should appear in metrics.json."""
    # Write predictions into the expected directory structure:
    # <results_dir>/<benchmark>/output.jsonl
    benchmark_dir = tmp_path / "wmt24"
    benchmark_dir.mkdir()
    output_file = benchmark_dir / "output.jsonl"
    with open(output_file, "w") as f:
        for record in PREDICTIONS:
            f.write(json.dumps(record) + "\n")

    summarize_results(results_dir=str(tmp_path), metric_type="translation")

    with open(tmp_path / "metrics.json") as f:
        metrics = json.load(f)

    # The top-level key is the benchmark directory name.
    assert "wmt24" in metrics, f"'wmt24' not in metrics.json keys: {list(metrics.keys())}"
    wmt24 = metrics["wmt24"]

    # All per-pair and aggregated keys from above should carry SEGALE scores.
    expected_keys = [
        "de_Latn->en_Latn",
        "zho_Hans->en_Latn",
        "xx->xx",
        "de_Latn->xx",
        "zho_Hans->xx",
        "xx->en_Latn",
    ]
    for lang_pair in expected_keys:
        assert lang_pair in wmt24, f"'{lang_pair}' missing from metrics.json"
        for field in SEGALE_FIELDS:
            assert field in wmt24[lang_pair], (
                f"'{field}' missing from metrics.json['{lang_pair}']. "
                "translation_metrics.py does not yet read SEGALE scores."
            )

    # Each document should contribute equally regardless of sentence count.
    # de_Latn->en_Latn: doc1=0.88 (3 sentences), doc2=0.90 (2 sentences).
    # Correct average = (0.88 + 0.90) / 2 = 0.89.
    # Wrong average (weighted by sentence count) = (0.88*3 + 0.90*2) / 5 = 0.888.
    assert abs(wmt24["de_Latn->en_Latn"]["segale_comet"] - 0.89) < 1e-9, (
        f"segale_comet for de_Latn->en_Latn is {wmt24['de_Latn->en_Latn']['segale_comet']}, "
        "expected 0.89. Each document should contribute equally, not weighted by sentence count."
    )


def test_segale_graceful_degradation_no_doc_id(tmp_path):
    """When doc_id is absent (sentence-level data), SEGALE fields should not appear.

    This tests the graceful degradation path: sentence-level benchmarks like
    FLORES-200 fall back to BLEU/COMET only without raising an error.
    """
    benchmark_dir = tmp_path / "flores200"
    benchmark_dir.mkdir()
    output_file = benchmark_dir / "output.jsonl"

    # Same records but without doc_id and without segale_* fields.
    sentence_level_records = [
        {k: v for k, v in r.items() if k not in ("doc_id",) + tuple(SEGALE_FIELDS)} for r in PREDICTIONS
    ]
    with open(output_file, "w") as f:
        for record in sentence_level_records:
            f.write(json.dumps(record) + "\n")

    summarize_results(results_dir=str(tmp_path), metric_type="translation")

    with open(tmp_path / "metrics.json") as f:
        metrics = json.load(f)

    flores200 = metrics["flores200"]
    # BLEU should still be present.
    assert "bleu" in flores200["de_Latn->en_Latn"]
    # SEGALE fields must NOT appear when doc_id / segale scores are absent.
    for field in SEGALE_FIELDS:
        assert field not in flores200["de_Latn->en_Latn"], (
            f"'{field}' should not appear in sentence-level benchmark results"
        )
