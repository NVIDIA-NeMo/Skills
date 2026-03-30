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

"""Unit tests for DocumentTranslationTask.

Tests preprocess_data and restore_async_order without a live model by:
  - bypassing GenerationTask.__init__ via __new__
  - writing a synthetic async file to simulate what the async loop produces
  - verifying the final output.jsonl has generation on seg_id=0 and "" on others
"""

import json

from nemo_skills.inference.document_translation import DocumentTranslationTask

SENTENCES = [
    # doc_1: 3 sentences, seg_ids 0-2
    {
        "doc_id": "doc_1",
        "seg_id": 0,
        "text": "First sentence.",
        "translation": "Erster Satz.",
        "source_language": "en",
        "target_language": "de_DE",
    },
    {
        "doc_id": "doc_1",
        "seg_id": 1,
        "text": "Second sentence.",
        "translation": "Zweiter Satz.",
        "source_language": "en",
        "target_language": "de_DE",
    },
    {
        "doc_id": "doc_1",
        "seg_id": 2,
        "text": "Third sentence.",
        "translation": "Dritter Satz.",
        "source_language": "en",
        "target_language": "de_DE",
    },
    # doc_2: 2 sentences, seg_ids 0-1
    {
        "doc_id": "doc_2",
        "seg_id": 0,
        "text": "Hello world.",
        "translation": "Hallo Welt.",
        "source_language": "en",
        "target_language": "de_DE",
    },
    {
        "doc_id": "doc_2",
        "seg_id": 1,
        "text": "Goodbye world.",
        "translation": "Auf Wiedersehen Welt.",
        "source_language": "en",
        "target_language": "de_DE",
    },
]


def make_task(tmp_path):
    """Instantiate DocumentTranslationTask without triggering GenerationTask.__init__."""

    class FakeCfg:
        output_file = str(tmp_path / "output.jsonl")
        input_file = str(tmp_path / "input.jsonl")
        async_position_key = "_async_position"
        generation_key = "generation"
        enable_litellm_cache = False
        target_languages = []
        skip_filled = False

    task = DocumentTranslationTask.__new__(DocumentTranslationTask)
    task.cfg = FakeCfg()
    return task


def test_preprocess_data_groups_by_doc_id(tmp_path):
    task = make_task(tmp_path)
    doc_records = task.preprocess_data(list(SENTENCES))

    assert len(doc_records) == 2, "Should produce one record per document"
    assert task._doc_order == [("doc_1", "de_DE"), ("doc_2", "de_DE")]

    # doc_1: three sentences joined with a space
    assert doc_records[0]["doc_id"] == "doc_1"
    assert doc_records[0]["text"] == "First sentence. Second sentence. Third sentence."

    # doc_2: two sentences joined with a space
    assert doc_records[1]["doc_id"] == "doc_2"
    assert doc_records[1]["text"] == "Hello world. Goodbye world."


def test_preprocess_data_sorts_by_seg_id(tmp_path):
    """Records arriving out of seg_id order should be sorted before concatenation."""
    task = make_task(tmp_path)
    shuffled = [SENTENCES[2], SENTENCES[0], SENTENCES[1], SENTENCES[4], SENTENCES[3]]
    doc_records = task.preprocess_data(shuffled)

    assert doc_records[0]["text"] == "First sentence. Second sentence. Third sentence."
    assert doc_records[1]["text"] == "Hello world. Goodbye world."


def test_preprocess_data_uses_first_sentence_fields(tmp_path):
    """Non-text fields on the doc record should come from seg_id=0."""
    task = make_task(tmp_path)
    doc_records = task.preprocess_data(list(SENTENCES))

    # translation field should be from seg_id=0, not concatenated
    assert doc_records[0]["translation"] == "Erster Satz."
    assert doc_records[0]["seg_id"] == 0


def test_restore_async_order_fans_out(tmp_path):
    """Full round-trip: preprocess → write fake async file → restore → verify output."""
    task = make_task(tmp_path)
    task.preprocess_data(list(SENTENCES))

    # Write the input file that restore_async_order reads to recover per-sentence records.
    with open(task.cfg.input_file, "w") as f:
        for s in SENTENCES:
            f.write(json.dumps(s) + "\n")

    # Simulate what the async loop would write: two doc-level outputs, out of order.
    doc1_generation = "Erster Satz. Zweiter Satz. Dritter Satz."
    doc2_generation = "Hallo Welt. Auf Wiedersehen Welt."

    async_records = [
        # doc_2 finishes first (position 1)
        {
            "generation": doc2_generation,
            "doc_id": "doc_2",
            "target_language": "de_DE",
            "text": "Hello world. Goodbye world.",
            "_async_position": 1,
        },
        # doc_1 finishes second (position 0)
        {
            "generation": doc1_generation,
            "doc_id": "doc_1",
            "target_language": "de_DE",
            "text": "First sentence. Second sentence. Third sentence.",
            "_async_position": 0,
        },
    ]

    async_path = task.cfg.output_file + "-async"
    with open(async_path, "w") as f:
        for rec in async_records:
            f.write(json.dumps(rec) + "\n")

    task.restore_async_order()

    assert not __import__("pathlib").Path(async_path).exists(), "async file should be deleted"

    with open(task.cfg.output_file) as f:
        output = [json.loads(line) for line in f]

    assert len(output) == 5, "Should have one record per original sentence"

    # doc_1 records: generation on seg_id=0, "" on others
    assert output[0]["doc_id"] == "doc_1"
    assert output[0]["seg_id"] == 0
    assert output[0]["generation"] == doc1_generation
    assert output[0]["text"] == "First sentence."  # original sentence text, not concatenated

    assert output[1]["seg_id"] == 1
    assert output[1]["generation"] == ""
    assert output[1]["text"] == "Second sentence."

    assert output[2]["seg_id"] == 2
    assert output[2]["generation"] == ""

    # doc_2 records
    assert output[3]["doc_id"] == "doc_2"
    assert output[3]["seg_id"] == 0
    assert output[3]["generation"] == doc2_generation

    assert output[4]["seg_id"] == 1
    assert output[4]["generation"] == ""


def test_restore_async_order_preserves_per_sentence_fields(tmp_path):
    """Each output record should have the original per-sentence translation field."""
    task = make_task(tmp_path)
    task.preprocess_data(list(SENTENCES))

    # Write the input file that restore_async_order reads to recover per-sentence records.
    with open(task.cfg.input_file, "w") as f:
        for s in SENTENCES:
            f.write(json.dumps(s) + "\n")

    async_records = [
        {
            "generation": "Erster Satz. Zweiter Satz. Dritter Satz.",
            "doc_id": "doc_1",
            "target_language": "de_DE",
            "_async_position": 0,
        },
        {
            "generation": "Hallo Welt. Auf Wiedersehen Welt.",
            "doc_id": "doc_2",
            "target_language": "de_DE",
            "_async_position": 1,
        },
    ]
    with open(task.cfg.output_file + "-async", "w") as f:
        for rec in async_records:
            f.write(json.dumps(rec) + "\n")

    task.restore_async_order()

    with open(task.cfg.output_file) as f:
        output = [json.loads(line) for line in f]

    # Each sentence record should have its own reference translation, not the first sentence's
    assert output[0]["translation"] == "Erster Satz."
    assert output[1]["translation"] == "Zweiter Satz."
    assert output[2]["translation"] == "Dritter Satz."
    assert output[3]["translation"] == "Hallo Welt."
    assert output[4]["translation"] == "Auf Wiedersehen Welt."
