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

"""DocumentTranslationTask: groups sentence records by doc_id, sends the full
document as a single natural-prose string to the model, then fans the result
back out to per-sentence records (full generation on seg_id=0, "" on rest).

This is designed for use with wmt24pp and any other benchmark that provides
doc_id and seg_id fields. SEGALE's judge script reads the output.jsonl and
expects this exact format.
"""

import json
import logging
import random
import re
import sys
from pathlib import Path

import hydra

from nemo_skills.code_execution.sandbox import sandbox_params
from nemo_skills.inference.generate import GenerationTask, GenerationTaskConfig
from nemo_skills.inference.model import server_params
from nemo_skills.utils import get_help_message, get_logger_name, setup_logging

LOG = logging.getLogger(get_logger_name(__file__))


class DocumentTranslationTask(GenerationTask):
    """GenerationTask subclass for document-level MT.

    Input: one record per source sentence (with doc_id and seg_id fields).

    Processing:
      - preprocess_data groups sentences by doc_id, joins source sentences with
        a space into natural prose, and presents each document as a single
        generation request.
      - restore_async_order fans the model's document-level output back out to
        per-sentence records: full generation stored on seg_id=0, "" on all
        subsequent records within the same document.

    This matches the format expected by segale_judge.py:
      seg_id=0: generation = full document translation string
      seg_id>0: generation = "" (ignored by SEGALE; source/reference used instead)
    """

    def skip_completed_samples(self, data):
        # Build a map from each sentence's index to its (doc_id, target_language)
        # key so that when a completed doc's sentinel position (sentence[0]'s idx)
        # is found in the async file, we can mark ALL sentences in that doc as
        # filled — not just sentence[0].  The base class only records sentence[0]'s
        # position in the async file, so without this override the other sentences
        # of already-translated docs would be re-submitted on resume.
        doc_to_indices = {}  # (doc_id, tgt_lang) -> sorted list of (seg_id, idx)
        for idx, record in enumerate(data):
            key = (record["doc_id"], record["target_language"])
            if key not in doc_to_indices:
                doc_to_indices[key] = []
            doc_to_indices[key].append((record["seg_id"], idx))
        for key in doc_to_indices:
            doc_to_indices[key].sort()  # sort by seg_id so index 0 is sentence[0]

        # Map from sentence[0]'s idx -> set of ALL indices in that doc.
        sentinel_to_all = {}
        for key, seg_idx_pairs in doc_to_indices.items():
            sentinel_idx = seg_idx_pairs[0][1]  # idx of the lowest-seg_id sentence
            all_indices = {idx for _, idx in seg_idx_pairs}
            sentinel_to_all[sentinel_idx] = all_indices

        # Let the base class read filled sentinel positions from the async file,
        # then expand each sentinel to cover every sentence in its document.
        remaining = super().skip_completed_samples(data)
        if not self.cfg.skip_filled:
            return remaining

        # Determine which sentinel positions were already filled.
        filled_sentinels = set()
        try:
            with open(self.cfg.output_file + "-async", "rt", encoding="utf-8") as fin:
                for line in fin:
                    pos = int(json.loads(line)[self.cfg.async_position_key])
                    if pos in sentinel_to_all:
                        filled_sentinels.add(pos)
        except FileNotFoundError:
            pass

        if not filled_sentinels:
            return remaining

        # Expand: all sentence indices belonging to filled docs.
        all_filled = set()
        for sentinel in filled_sentinels:
            all_filled |= sentinel_to_all[sentinel]

        # Re-filter: keep only records whose original index is not fully filled.
        return [dp for dp in remaining if dp[self.cfg.async_position_key] not in all_filled]

    def preprocess_data(self, data):
        # Filter by target_languages if specified.
        target_languages = self.cfg.target_languages
        if target_languages:
            available_languages = {r["target_language"] for r in data if "target_language" in r}
            missing_languages = sorted(set(target_languages) - available_languages)
            if missing_languages:
                raise ValueError(
                    f"Unsupported target_languages={missing_languages}; "
                    f"available languages: {sorted(available_languages)}"
                )
            data = [r for r in data if r["target_language"] in target_languages]
            if not data:
                raise ValueError(f"No records found for target_languages={target_languages}")
            LOG.info("Filtered to %d records for languages: %s", len(data), target_languages)

        # Group sentence records by (doc_id, target_language), preserving document
        # order. Keying on both fields is necessary when a single test.jsonl contains
        # multiple languages that share the same doc_id (e.g. wmt24pp.doc).
        docs = {}  # (doc_id, target_language) -> list of sentence records
        for record in data:
            key = (record["doc_id"], record["target_language"])
            if key not in docs:
                docs[key] = []
            docs[key].append(record)

        # Sort each document's sentences by seg_id.
        for key in docs:
            docs[key].sort(key=lambda r: r["seg_id"])

        # Save for use in restore_async_order.
        self._doc_records = docs
        self._doc_order = list(docs.keys())

        # Build one record per document.
        # Join source sentences with a space so the input reads as natural prose.
        # Newlines would give the model artificial sentence boundary hints,
        # defeating SEGALE's alignment step.
        #
        # For long-context records (identified by a "doc_groups" field), shuffle
        # document order when the output file carries a random-seed suffix (-rsN).
        # This creates statistically independent runs across seeds while keeping
        # canonical ordering for the default single-seed case.
        is_seeded_run = bool(re.search(r"-rs\d+", Path(self.cfg.output_file).name))

        doc_records = []
        for key in self._doc_order:
            sentences = docs[key]
            doc_record = dict(sentences[0])

            if is_seeded_run and "doc_groups" in doc_record:
                seed = self.cfg.inference.random_seed
                groups = list(doc_record["doc_groups"])
                random.Random(seed).shuffle(groups)
                doc_record["source_sentences"] = [s for g in groups for s in g["source_sentences"]]
                doc_record["reference_sentences"] = [s for g in groups for s in g["reference_sentences"]]
                doc_record["text"] = " ".join(doc_record["source_sentences"])
            else:
                doc_record["text"] = " ".join(r["text"] for r in sentences)

            doc_records.append(doc_record)

        return doc_records

    def restore_async_order(self):
        # Read document-level outputs from the async file.
        with open(self.cfg.output_file + "-async", "rt", encoding="utf-8") as fin:
            doc_outputs = [json.loads(line) for line in fin]

        # Sort by async position to recover document order.
        # Positions are inherited from the original sentence-level indices set by
        # skip_completed_samples() before preprocess_data() runs, so they are not
        # necessarily 0..N-1 — sort by value rather than using as a direct index.
        ordered = sorted(doc_outputs, key=lambda d: d[self.cfg.async_position_key])
        for d in ordered:
            d.pop(self.cfg.async_position_key)

        generation_key = self.cfg.generation_key

        # Rebuild sentence-level records from the original input file.
        # We cannot rely on self._doc_records/_doc_order here because preprocess_data
        # is called with only the *remaining* (unfinished) documents — on a resume run
        # those dicts cover a subset of the async file, causing an IndexError when
        # restore_async_order tries to index beyond len(_doc_order).
        with open(self.cfg.input_file, "rt", encoding="utf-8") as fin:
            all_sentences = [json.loads(line) for line in fin if line.strip()]

        input_docs = {}
        for record in all_sentences:
            key = (record["doc_id"], record["target_language"])
            if key not in input_docs:
                input_docs[key] = []
            input_docs[key].append(record)
        for key in input_docs:
            input_docs[key].sort(key=lambda r: r["seg_id"])

        # Fan out: one document output -> N per-sentence records.
        with open(self.cfg.output_file, "wt", encoding="utf-8") as fout:
            for doc_output in ordered:
                key = (doc_output["doc_id"], doc_output["target_language"])
                sentences = input_docs[key]
                doc_generation = doc_output[generation_key]

                for i, sentence in enumerate(sentences):
                    # Start from the original per-sentence record so that
                    # fields like text, seg_id, translation are preserved correctly.
                    record = dict(sentence)
                    record[generation_key] = doc_generation if i == 0 else ""
                    # For long-context records, preprocess_data may have shuffled
                    # doc_groups and written shuffled text/source_sentences/reference_sentences
                    # into the async record. Propagate them so SEGALE embeds source text in
                    # the same order as the model's actual input.
                    # "text" is only propagated when the fan-out is 1-to-1 (one sentence per
                    # doc): in the sentence-level fan-out case each sentence has its own
                    # per-sentence source text that must not be overwritten with the full
                    # concatenated doc text.
                    if i == 0:
                        fields = ("source_sentences", "reference_sentences")
                        if len(sentences) == 1:
                            fields = ("text",) + fields
                        for field in fields:
                            if field in doc_output:
                                record[field] = doc_output[field]
                    fout.write(json.dumps(record) + "\n")

        Path(self.cfg.output_file + "-async").unlink()
        self.cleanup_litellm_cache()

    def generate(self):
        super().generate()
        # If the async file still exists after generate(), it means skip_completed_samples
        # returned empty (all docs already done), async_loop was skipped entirely, and
        # restore_async_order was never called. Reconstruct and fan out now.
        async_file = Path(self.cfg.output_file + "-async")
        if async_file.exists():
            LOG.info(
                "Async file present after generate(); all docs were pre-filled. "
                "Calling restore_async_order to complete fan-out."
            )
            self.restore_async_order()


GENERATION_TASK_CLASS = DocumentTranslationTask


@hydra.main(version_base=None, config_name="base_generation_config")
def generate(cfg: GenerationTaskConfig):
    cfg = GenerationTaskConfig(_init_nested=True, **cfg)
    LOG.info("Config used: %s", cfg)
    task = DocumentTranslationTask(cfg)
    task.generate()


HELP_MESSAGE = get_help_message(
    GenerationTaskConfig,
    server_params=server_params(),
    sandbox_params=sandbox_params(),
)

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(HELP_MESSAGE)
    else:
        setup_logging()
        generate()
