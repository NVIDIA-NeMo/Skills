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

"""
SEGALE judge script for document-level machine translation evaluation.

Runs as a 3-phase disaggregated pipeline invoked with --mode:

  --mode embed  (GPU job)
    Segments source and generation text with ersatz, computes LASER2
    embeddings, and saves sentences/overlaps/embeddings to disk.
    Source embeddings are computed once and shared across all target languages.
    Writes segale_intermediate/embed/embed.done on success.

  --mode align  (CPU job, no GPU needed)
    Loads pre-computed embeddings from the embed phase and runs vecalign
    alignment in parallel across all target languages using all available CPUs.
    Writes per-language aligned_spans.jsonl and align.done files.

  --mode score  (GPU job)
    Loads all aligned spans across all languages and batch-scores them with
    COMET-QE (COMETKiwi wmt22) and MetricX-QE in a single GPU pass.
    Writes per-language output.jsonl files under per_lang/<lang>/.

  --merge-input / --merge-langs-dir  (CPU, no mode flag)
    Reassembles per-language scored outputs into the final output.jsonl
    in the original record order.

Output fields per document: segale_comet_qe, segale_metricx_qe,
                             segale_lang_fidelity, segale_total_seg,
                             segale_misaligned_seg

Score sentinels (from segale_eval)
-----------------------------------
segale_eval's scoring functions return sentinel values for spans that cannot
be meaningfully scored due to alignment edge cases:

  -1  Both src and tgt are absent/empty.  Returned for completely invalid
      windows.  Extremely rare in practice — vecalign almost always produces
      at least one non-empty side.  Excluded from the doc-level average
      (filtered by >= 0).

   0  Exactly one of src or tgt is an empty string.  Two causes:
        deleted      — tgt=""  (src_indices produced content but mt_indices=[])
                       The model skipped translating this source span.
        hallucinated — src=""  (mt_indices produced content but src_indices=[])
                       The model generated content with no corresponding source.
      Score 0 IS included in the doc-level average, so deletions and
      hallucinations directly lower segale_comet_qe and segale_metricx_qe.

A span is "misaligned" (counted in segale_misaligned_seg) when its score
is <= 0, i.e. it is either deleted, hallucinated, or completely invalid.
In spans.jsonl each span carries explicit "deleted" and "hallucinated" flags
(derived from empty tgt / empty src respectively) so the two cases can be
distinguished without relying on the score sentinel.

Model downloads
---------------
  * LASER2 (alignment embeddings)
      Downloaded by laser_encoders to $LASER_HOME or ~/.cache/laser_encoders/

  * COMETKiwi (scoring)
      Downloaded by unbabel-comet to $HF_HOME/hub/
      Model: Unbabel/wmt22-cometkiwi-da

  * MetricX-QE (scoring)
      Downloaded by HuggingFace transformers to $HF_HOME/hub/
      Model: google/metricx-24-hybrid-large-v2p6

On a cluster with a shared parallel filesystem set HF_HOME to a shared path
so model weights are downloaded once and reused across all jobs.
"""

import argparse
import json
import logging
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOG = logging.getLogger(__name__)


def _clean_sentences(sentences: list[str]) -> list[str]:
    """Replace line-break characters within sentences with spaces.

    Newlines and carriage returns inside a sentence would corrupt vecalign's
    overlap file (which is newline-delimited), splitting one entry into
    multiple lines. SEGALE's generate_overlap_and_embedding uses set() to
    deduplicate overlaps internally, so no explicit dedup is needed here.
    """
    return [s.replace("\r", " ").replace("\n", " ") for s in sentences]


def _compute_lang_fidelity(
    generation: str,
    reference_text: str,
    chunk_size: int = 500,
) -> float:
    """Compute the fraction of generation chunks in the correct target language.

    Detects the expected target language from the reference text, then checks
    each chunk of the generation against it. Returns 1.0 if all chunks match
    the target language, 0.0 if none do.

    This catches models that collapse to the source language (e.g. English)
    or any other wrong language mid-generation.

    Returns 1.0 for empty/short text or if langdetect is not installed.
    """
    if not generation or len(generation) < 50:
        return 1.0

    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
    except ImportError:
        LOG.warning("langdetect not installed — skipping language fidelity check.")
        return 1.0

    # Detect expected target language from the reference
    try:
        expected_lang = detect(reference_text[:10000])
    except Exception:
        LOG.warning("Could not detect reference language — skipping fidelity check.")
        return 1.0

    chunks = [generation[i : i + chunk_size] for i in range(0, len(generation), chunk_size)]
    if len(chunks) > 1 and len(chunks[-1]) < 100:
        chunks = chunks[:-1]
    if not chunks:
        return 1.0

    correct_count = 0
    detected_count = 0
    for chunk in chunks:
        try:
            detected = detect(chunk)
            detected_count += 1
            if detected == expected_lang:
                correct_count += 1
        except Exception:
            pass  # undetectable chunk — excluded from ratio

    if detected_count == 0:
        return 1.0
    return correct_count / detected_count


# ====================================================================
# 3-Phase disaggregated pipeline: Embed → Align → Score
# ====================================================================


def _find_ersatz_model_path() -> str | None:
    """Return the path to the ersatz multilingual model checkpoint.

    Looks inside $ERSATZ for the first *.multilingual file (the pattern
    used by the default-multilingual model downloaded by the cluster setup).
    Returns None if ERSATZ is unset or no model is found.
    """
    import glob

    ersatz_home = os.environ.get("ERSATZ", "")
    if not ersatz_home:
        return None
    for pattern in [
        os.path.join(ersatz_home, "**", "*.multilingual"),
        os.path.join(ersatz_home, "**", "*multilingual"),
    ]:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            return matches[0]
    return None


def _batch_segment_ersatz(texts: list[str]) -> list[list[str]]:
    """Segment all texts in one ersatz subprocess call (one model load total).

    Writes all texts to a single temp file separated by a unique ASCII marker
    line, calls the ersatz CLI once, then splits the output on that marker to
    recover per-document sentence lists.  One subprocess = one model load,
    and ersatz's internal batching handles GPU throughput.

    A canary document is prepended to the batch so the marker behavior can be
    verified in-band before any real output is trusted.  If the canary check
    or the marker count check fails, falls back to per-text subprocess calls.
    """
    import subprocess
    import tempfile
    import time

    # Plain ASCII, no punctuation — ersatz finds no sentence boundary in it.
    MARKER = "SEGALE-DOC-SEP-XXXXXXXXXXXXXXXXXXX"

    # Canary: a known two-sentence doc prepended to position 0.
    # After running ersatz we verify it produced exactly two sentences before
    # the first marker, confirming the marker is acting as a clean boundary.
    CANARY_TEXT = "The canary document has exactly two sentences. This is the second one."
    CANARY_EXPECTED = {"The canary document has exactly two sentences.", "This is the second one."}

    all_texts = [CANARY_TEXT] + texts
    n_chars = sum(len(t) for t in all_texts)

    with tempfile.NamedTemporaryFile(delete=False, mode="w", encoding="utf-8", suffix=".txt") as fh:
        for text in all_texts:
            fh.write(text.strip())
            fh.write("\n" + MARKER + "\n")
        input_path = fh.name

    output_path = input_path + ".segmented"

    def _fallback():
        LOG.warning(
            "ersatz batch: marker self-test failed — falling back to per-text subprocess to avoid boundary corruption"
        )
        import segale_align as sa

        return [_clean_sentences(sa.segment_sentences_by_ersatz(t)) for t in texts]

    LOG.info(
        "ersatz batch: calling subprocess on %d texts (%d chars) via %s",
        len(all_texts),
        n_chars,
        input_path,
    )
    t0 = time.monotonic()
    try:
        subprocess.run(
            ["ersatz", "--input", input_path, "--output", output_path],
            check=True,
        )
    finally:
        os.remove(input_path)
    elapsed = time.monotonic() - t0
    LOG.info("ersatz batch: subprocess finished in %.1fs", elapsed)

    with open(output_path, "r", encoding="utf-8") as fh:
        output_lines = [line.rstrip("\n") for line in fh]
    os.remove(output_path)

    LOG.info(
        "ersatz batch: output has %d lines for %d input texts",
        len(output_lines),
        len(all_texts),
    )

    # Expect one marker per input text (including the canary).
    # l.strip() == MARKER is True only when the entire line (minus whitespace)
    # is the marker — if ersatz merged the marker with adjacent text, l.strip()
    # would contain extra content and NOT equal MARKER, reducing the count and
    # triggering the fallback.
    markers_in_output = sum(1 for line in output_lines if line.strip() == MARKER)
    if markers_in_output != len(all_texts):
        LOG.warning(
            "ersatz batch: expected %d isolated marker lines in output, got %d "
            "(ersatz may have merged or dropped the marker) — falling back",
            len(all_texts),
            markers_in_output,
        )
        return _fallback()

    # Split on markers.
    raw_results: list[list[str]] = []
    current: list[str] = []
    for line in output_lines:
        if line.strip() == MARKER:
            raw_results.append(current)
            current = []
        elif line.strip():
            current.append(line.strip())

    # Verify canary (position 0): ersatz must have produced its two known sentences
    # and nothing else before the first marker.
    canary_out = set(raw_results[0]) if raw_results else set()
    if canary_out != CANARY_EXPECTED:
        LOG.warning(
            "ersatz batch: canary check failed (got %s, expected %s) — falling back",
            canary_out,
            CANARY_EXPECTED,
        )
        return _fallback()

    results = [_clean_sentences(sents) for sents in raw_results[1:]]
    total_sents = sum(len(s) for s in results)
    LOG.info(
        "ersatz batch: canary OK — %d texts → %d sentences (avg %.1f per doc)",
        len(results),
        total_sents,
        total_sents / len(results) if results else 0,
    )
    return results


def _run_embed_phase(
    input_file: Path,
    output_dir: Path,
    segmenter: str,
    judge_debug: bool = False,
    target_languages: set[str] | None = None,
    embed_batch_size: int = 512,
):
    """Phase 1 (GPU): Segment all docs and compute LASER2 embeddings.

    Two-pass approach:
      Pass 1 (GPU): batch-segment all docs with ersatz (EvalModel loaded once, one GPU
                    pass over all text), then generate overlap windows via stub encoder.
      Pass 2 (GPU): encode all overlaps in one flat batched call to model.encode_sentences(),
                    then slice results back into per-doc numpy arrays.

    Source embeddings are computed once per doc_id and shared across all target languages.

    Output layout in <output_dir>/segale_intermediate/embed/:
        doc_records.jsonl                 — serialized input doc records
        sentences/src/<doc_id>.json       — {"sentences": [...]}
        sentences/mt/<lang>/<doc_id>.json
        overlaps/src/<doc_id>.json        — {"overlaps": [...]}
        overlaps/mt/<lang>/<doc_id>.json
        embeddings/src/<doc_id>.npy       — float32 (n_overlaps, embed_dim)
        embeddings/mt/<lang>/<doc_id>.npy
        embed.done                        — sentinel written on success
    """
    import numpy as np

    inter_dir = output_dir / "segale_intermediate" / "embed"
    done_file = inter_dir / "embed.done"
    if done_file.exists():
        LOG.info("embed.done exists at %s — skipping embed phase", done_file)
        return

    inter_dir.mkdir(parents=True, exist_ok=True)

    with open(input_file, "rt", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    doc_records = [r for r in records if "doc_id" in r]
    if target_languages:
        doc_records = [r for r in doc_records if r.get("target_language") in target_languages]

    if judge_debug:
        debug_doc_ids = list(dict.fromkeys(r["doc_id"] for r in doc_records))[:2]
        doc_records = [r for r in doc_records if r["doc_id"] in debug_doc_ids]
        LOG.info("--judge-debug: limiting embed phase to %d docs (%s)", len(debug_doc_ids), debug_doc_ids)

    if not doc_records:
        LOG.warning("embed: no doc records found in %s", input_file)
        done_file.touch()
        return

    # Save doc_records for downstream align and score phases.
    with open(inter_dir / "doc_records.jsonl", "wt", encoding="utf-8") as f:
        for r in doc_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    try:
        import segale_align as sa
    except ImportError as e:
        raise ImportError("segale_align could not be imported. Ensure the SEGALE package is installed.") from e

    sa.VERBOSE = 2 if judge_debug else 0
    sa.SPACY = segmenter
    sa.STOP_JUMP = 0.15
    sa.COST_MIN = 0.30
    sa.COST_MAX = 0.30

    # ------------------------------------------------------------------ #
    # Pass 1a (GPU): batch-segment ALL texts with ersatz in one GPU pass  #
    # ------------------------------------------------------------------ #

    lang_doc_map: dict[str, dict[str, dict]] = defaultdict(dict)
    for record in doc_records:
        lang = record.get("target_language", "unknown")
        lang_doc_map[lang][record["doc_id"]] = record

    # Collect unique source texts and all MT texts.
    seen_src: set[str] = set()
    src_doc_ids: list[str] = []  # ordered unique doc_ids for source
    src_texts: list[str] = []
    for record in doc_records:
        doc_id = record["doc_id"]
        if doc_id not in seen_src:
            seen_src.add(doc_id)
            src_doc_ids.append(doc_id)
            src_texts.append(record.get("text", ""))

    mt_keys: list[tuple[str, str]] = []  # (lang, doc_id)
    mt_texts: list[str] = []
    for lang, doc_map in lang_doc_map.items():
        for doc_id, record in doc_map.items():
            # Use a placeholder for empty generations so the pair still flows
            # through embed → align → score and receives a legitimate (very low)
            # COMET-QE score rather than being silently dropped. The placeholder
            # is a nonsense string that segments to exactly one sentence and
            # scores near zero against any real source.
            tgt_text = record.get("generation", "").strip() or "___null_translation___"
            mt_keys.append((lang, doc_id))
            mt_texts.append(tgt_text)

    if segmenter != "spacy":
        all_sentence_lists = _batch_segment_ersatz(src_texts + mt_texts)
    else:
        all_sentence_lists = [_clean_sentences(sa.segment_sentences_by_spacy(t)) for t in src_texts + mt_texts]

    src_sentences_map: dict[str, list[str]] = {
        doc_id: _clean_sentences(sents) for doc_id, sents in zip(src_doc_ids, all_sentence_lists[: len(src_doc_ids)])
    }
    mt_sentences_map: dict[tuple[str, str], list[str]] = {
        key: _clean_sentences(sents) for key, sents in zip(mt_keys, all_sentence_lists[len(src_doc_ids) :])
    }

    LOG.info(
        "embed pass 1a: segmented %d source docs + %d MT pairs",
        len(src_doc_ids),
        len(mt_keys),
    )

    # ------------------------------------------------------------------ #
    # Pass 1b: overlap generation via stub encoder                        #
    #                                                                      #
    # generate_overlap_and_embedding does internal sub-sentence splitting  #
    # we cannot replicate independently, so we feed it a fast stub that   #
    # returns a throwaway scalar. All CPU logic runs, we get the correct   #
    # overlap strings, and the GPU is untouched until pass 2.             #
    # ------------------------------------------------------------------ #

    class _StubEncoder:
        """Returns a throwaway scalar so generate_overlap_and_embedding runs
        its full CPU logic without touching the GPU."""

        def encode_sentences(self, sentences):
            return np.empty(1, dtype=np.float32)

    stub = _StubEncoder()

    sents_src_dir = inter_dir / "sentences" / "src"
    overlaps_src_dir = inter_dir / "overlaps" / "src"
    embeds_src_dir = inter_dir / "embeddings" / "src"
    for d in (sents_src_dir, overlaps_src_dir, embeds_src_dir):
        d.mkdir(parents=True, exist_ok=True)

    # overlap_index maps (side, lang_or_None, doc_id) -> (flat_start, flat_end)
    all_overlap_strings: list[str] = []
    overlap_index: dict[tuple, tuple[int, int]] = {}

    # -- Source overlaps --
    for doc_id in src_doc_ids:
        src_sentences = src_sentences_map[doc_id]
        (sents_src_dir / f"{doc_id}.json").write_text(json.dumps({"sentences": src_sentences}, ensure_ascii=False))

        if not src_sentences:
            LOG.warning("embed: empty source sentences for doc_id=%s", doc_id)
            continue

        src_overlaps, _ = sa.generate_overlap_and_embedding(src_sentences, stub, None, max_size=8)
        (overlaps_src_dir / f"{doc_id}.json").write_text(json.dumps({"overlaps": src_overlaps}, ensure_ascii=False))

        start = len(all_overlap_strings)
        all_overlap_strings.extend(src_overlaps)
        overlap_index[("src", None, doc_id)] = (start, len(all_overlap_strings))

    # -- MT overlaps --
    for lang, doc_map in lang_doc_map.items():
        sents_mt_dir = inter_dir / "sentences" / "mt" / lang
        overlaps_mt_dir = inter_dir / "overlaps" / "mt" / lang
        embeds_mt_dir = inter_dir / "embeddings" / "mt" / lang
        for d in (sents_mt_dir, overlaps_mt_dir, embeds_mt_dir):
            d.mkdir(parents=True, exist_ok=True)

        for doc_id, record in doc_map.items():
            key = (lang, doc_id)
            if key not in mt_sentences_map:
                LOG.warning("embed: empty generation for doc_id=%s lang=%s — skipping", doc_id, lang)
                continue

            mt_sentences = mt_sentences_map[key]
            (sents_mt_dir / f"{doc_id}.json").write_text(json.dumps({"sentences": mt_sentences}, ensure_ascii=False))

            if not mt_sentences:
                LOG.warning("embed: empty MT sentences for doc_id=%s lang=%s — skipping", doc_id, lang)
                continue

            mt_overlaps, _ = sa.generate_overlap_and_embedding(mt_sentences, stub, None, max_size=8)
            (overlaps_mt_dir / f"{doc_id}.json").write_text(json.dumps({"overlaps": mt_overlaps}, ensure_ascii=False))

            start = len(all_overlap_strings)
            all_overlap_strings.extend(mt_overlaps)
            overlap_index[("mt", lang, doc_id)] = (start, len(all_overlap_strings))

    n_mt_pairs = sum(len(d) for d in lang_doc_map.values())
    LOG.info(
        "embed pass 1b complete: %d total overlaps across %d source docs + %d MT pairs",
        len(all_overlap_strings),
        len(seen_src),
        n_mt_pairs,
    )

    if not all_overlap_strings:
        LOG.warning("embed: no overlaps generated — nothing to embed")
        done_file.touch()
        return

    # ------------------------------------------------------------------ #
    # Pass 2 (GPU): batch-encode all overlaps, slice back to per-doc .npy #
    # ------------------------------------------------------------------ #

    try:
        from laser_encoders import LaserEncoderPipeline
    except ImportError as e:
        raise ImportError("laser_encoders is required for the embed phase.") from e

    laser_home = os.environ.get("LASER_HOME")
    LOG.info("embed pass 2: loading LASER2 model from %s", laser_home)
    model = LaserEncoderPipeline(laser="laser2", model_dir=laser_home)

    n_total = len(all_overlap_strings)

    # Sort overlaps by character length so each batch contains similarly-sized
    # sequences.  This minimises padding waste inside LASER2's BiLSTM encoder
    # and improves GPU utilisation.  We restore the original order afterwards.
    # Sorting is deterministic: same input → same sort_idx on every run.
    sort_idx = np.argsort([len(s) for s in all_overlap_strings])
    restore_idx = np.empty(n_total, dtype=np.int64)
    restore_idx[sort_idx] = np.arange(n_total)
    sorted_overlaps = [all_overlap_strings[i] for i in sort_idx]

    sorted_embeddings = np.empty((n_total, 1024), dtype=np.float32)

    # Warm restart: if a checkpoint exists from a previous timed-out run, load
    # it and resume encoding from where it left off.  The sort order is
    # deterministic so no index file is needed — n_encoded is derived from the
    # saved array shape.
    checkpoint_file = inter_dir / "embed_checkpoint.npy"
    n_encoded = 0
    if checkpoint_file.exists():
        try:
            saved = np.load(str(checkpoint_file))
            n_encoded = saved.shape[0]
            sorted_embeddings[:n_encoded] = saved
            del saved
            LOG.info(
                "embed pass 2: warm restart — resuming from %d / %d overlaps (%.1f%%)",
                n_encoded,
                n_total,
                100.0 * n_encoded / n_total,
            )
        except Exception as exc:
            LOG.warning("embed pass 2: checkpoint load failed (%s) — encoding from scratch", exc)
            n_encoded = 0

    # Save a checkpoint every ~10% of total overlaps so a timeout loses at most
    # ~10% of work on the next restart.
    checkpoint_every = max(embed_batch_size, n_total // 10)
    next_checkpoint_at = ((n_encoded // checkpoint_every) + 1) * checkpoint_every

    LOG.info(
        "embed pass 2: encoding %d overlaps in batches of %d (length-sorted, checkpoint every %d)",
        n_total - n_encoded,
        embed_batch_size,
        checkpoint_every,
    )

    for batch_start in range(n_encoded, n_total, embed_batch_size):
        batch_end = min(batch_start + embed_batch_size, n_total)
        sorted_embeddings[batch_start:batch_end] = model.encode_sentences(sorted_overlaps[batch_start:batch_end])
        LOG.info(
            "embed pass 2: encoded %d / %d overlaps (%.1f%%)",
            batch_end,
            n_total,
            100.0 * batch_end / n_total,
        )
        if batch_end >= next_checkpoint_at:
            np.save(str(checkpoint_file), sorted_embeddings[:batch_end])
            LOG.info("embed pass 2: checkpoint saved at %d / %d overlaps", batch_end, n_total)
            next_checkpoint_at = ((batch_end // checkpoint_every) + 1) * checkpoint_every

    # Restore original order so overlap_index slicing stays correct.
    all_embeddings = sorted_embeddings[restore_idx]
    LOG.info("embed pass 2: encoding complete, slicing back to per-doc arrays")

    # -- Slice source embeddings --
    for doc_id in seen_src:
        key = ("src", None, doc_id)
        if key not in overlap_index:
            continue
        start, end = overlap_index[key]
        np.save(str(embeds_src_dir / f"{doc_id}.npy"), all_embeddings[start:end])

    # -- Slice MT embeddings --
    for lang, doc_map in lang_doc_map.items():
        embeds_mt_dir = inter_dir / "embeddings" / "mt" / lang
        for doc_id in doc_map:
            key = ("mt", lang, doc_id)
            if key not in overlap_index:
                continue
            start, end = overlap_index[key]
            np.save(str(embeds_mt_dir / f"{doc_id}.npy"), all_embeddings[start:end])

    # Remove checkpoint now that all per-doc .npy files are written.
    if checkpoint_file.exists():
        checkpoint_file.unlink()
        LOG.info("embed pass 2: checkpoint removed")

    done_file.touch()
    LOG.info("Embed phase complete. Sentinel: %s", done_file)


def _align_lang_worker(work_item: dict) -> tuple[str, int]:
    """Worker function for ProcessPoolExecutor: align one target language.

    Loads pre-computed embeddings and overlaps from the embed phase, runs
    vecalign per document, and writes aligned_spans.jsonl plus align.done.
    Must be module-level so it is picklable by ProcessPoolExecutor.

    Returns (lang, n_spans) on success; raises on failure.
    """
    import json as _json
    import logging as _logging
    import os as _os
    from pathlib import Path as _Path

    import numpy as _np

    _log = _logging.getLogger(__name__)

    lang: str = work_item["lang"]
    doc_records: list[dict] = work_item["doc_records"]
    embed_dir = _Path(work_item["embed_dir"])
    align_lang_dir = _Path(work_item["align_lang_dir"])
    segmenter: str = work_item["segmenter"]
    judge_debug: bool = work_item["judge_debug"]

    done_file = align_lang_dir / "align.done"
    if done_file.exists():
        _log.info("align: lang=%s already done — skipping", lang)
        return lang, 0

    align_lang_dir.mkdir(parents=True, exist_ok=True)

    try:
        import segale_align as sa
    except ImportError as e:
        raise ImportError("segale_align could not be imported in align worker.") from e

    sa.VERBOSE = 2 if judge_debug else 0
    sa.SPACY = segmenter
    sa.STOP_JUMP = 0.15
    sa.COST_MIN = 0.30
    sa.COST_MAX = 0.30

    all_spans: list[dict] = []
    for record in doc_records:
        doc_id = record["doc_id"]

        src_sents_file = embed_dir / "sentences" / "src" / f"{doc_id}.json"
        src_overlap_file = embed_dir / "overlaps" / "src" / f"{doc_id}.json"
        src_embed_file = embed_dir / "embeddings" / "src" / f"{doc_id}.npy"

        if not src_sents_file.exists() or not src_embed_file.exists():
            _log.warning("align: missing source data for doc_id=%s lang=%s — skipping", doc_id, lang)
            continue

        src_sentences = _json.loads(src_sents_file.read_text(encoding="utf-8"))["sentences"]
        src_overlap = _json.loads(src_overlap_file.read_text(encoding="utf-8"))["overlaps"]
        src_embed = _np.load(str(src_embed_file))

        mt_sents_file = embed_dir / "sentences" / "mt" / lang / f"{doc_id}.json"
        mt_overlap_file = embed_dir / "overlaps" / "mt" / lang / f"{doc_id}.json"
        mt_embed_file = embed_dir / "embeddings" / "mt" / lang / f"{doc_id}.npy"

        if not mt_sents_file.exists() or not mt_embed_file.exists():
            _log.warning("align: missing MT data for doc_id=%s lang=%s — skipping", doc_id, lang)
            continue

        mt_sentences = _json.loads(mt_sents_file.read_text(encoding="utf-8"))["sentences"]
        mt_overlap = _json.loads(mt_overlap_file.read_text(encoding="utf-8"))["overlaps"]
        mt_embed = _np.load(str(mt_embed_file))

        # Each doc gets its own temp folder to avoid vecalign file collisions.
        save_folder = str(align_lang_dir / "vecalign_tmp" / doc_id)
        _os.makedirs(save_folder, exist_ok=True)

        src_mt_alignments = sa.run_vecalign_explore(
            "\n".join(src_sentences),
            "\n".join(mt_sentences),
            "\n".join(src_overlap),
            "\n".join(mt_overlap),
            src_embed,
            mt_embed,
            doc_id,
            save_folder,
            max_size=8,
        )

        for seg_id, (src_indices, mt_indices) in enumerate(src_mt_alignments, start=1):
            aligned_src = " ".join([src_sentences[i] for i in src_indices]) if src_indices else ""
            aligned_mt = " ".join([mt_sentences[i] for i in mt_indices]) if mt_indices else ""
            all_spans.append(
                {
                    "doc_id": doc_id,
                    "src": aligned_src,
                    "tgt": aligned_mt,
                    "seg_id": seg_id,
                }
            )

    with open(align_lang_dir / "aligned_spans.jsonl", "wt", encoding="utf-8") as fh:
        for span in all_spans:
            fh.write(_json.dumps(span, ensure_ascii=False) + "\n")

    done_file.touch()
    _log.info("align: lang=%s done (%d spans)", lang, len(all_spans))
    return lang, len(all_spans)


def _run_align_phase(
    output_dir: Path,
    segmenter: str,
    judge_debug: bool = False,
    target_languages: set[str] | None = None,
):
    """Phase 2 (CPU): Run vecalign alignment in parallel across all languages.

    Reads pre-computed embeddings from the embed phase intermediate directory,
    runs vecalign for every (language, doc_id) pair using all available CPUs,
    and writes per-language aligned_spans.jsonl files.

    No GPU is required.  Set CUDA_VISIBLE_DEVICES="" before invoking to prevent
    torch from attempting to initialize the CUDA runtime.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    inter_dir = output_dir / "segale_intermediate"
    embed_dir = inter_dir / "embed"

    if not (embed_dir / "embed.done").exists():
        LOG.error("embed.done not found at %s — run embed phase first", embed_dir)
        sys.exit(1)

    with open(embed_dir / "doc_records.jsonl", "rt", encoding="utf-8") as f:
        doc_records = [json.loads(line) for line in f if line.strip()]

    lang_records: dict[str, list] = defaultdict(list)
    for r in doc_records:
        lang = r.get("target_language", "unknown")
        lang_records[lang].append(r)

    if target_languages:
        lang_records = {lang: rs for lang, rs in lang_records.items() if lang in target_languages}

    all_langs = sorted(lang_records.keys())
    langs_to_process = [lang for lang in all_langs if not (inter_dir / "align" / lang / "align.done").exists()]
    skipped = len(all_langs) - len(langs_to_process)
    if skipped:
        LOG.info("align: %d languages already complete — skipping", skipped)

    if not langs_to_process:
        LOG.info("align: all %d languages already complete", len(all_langs))
        return

    n_workers = min(os.cpu_count() or 1, len(langs_to_process))
    LOG.info("align: processing %d languages with %d CPU workers", len(langs_to_process), n_workers)

    work_items = [
        {
            "lang": lang,
            "doc_records": lang_records[lang],
            "embed_dir": str(embed_dir),
            "align_lang_dir": str(inter_dir / "align" / lang),
            "segmenter": segmenter,
            "judge_debug": judge_debug,
        }
        for lang in langs_to_process
    ]

    failed: list[str] = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_lang = {executor.submit(_align_lang_worker, item): item["lang"] for item in work_items}
        for future in as_completed(future_to_lang):
            lang = future_to_lang[future]
            try:
                _, n_spans = future.result()
                LOG.info("align: lang=%s finished (%d spans)", lang, n_spans)
            except Exception as exc:
                LOG.error("align: lang=%s failed: %s", lang, exc)
                failed.append(lang)

    if failed:
        LOG.error("Align phase failed for: %s", failed)
        sys.exit(1)

    LOG.info("Align phase complete for all %d languages", len(langs_to_process))


def _run_score_phase(
    input_file: Path,
    output_dir: Path,
    target_languages: set[str] | None = None,
    save_spans: bool = False,
):
    """Phase 3 (GPU): Batch-score aligned spans with COMET-QE and MetricX-QE.

    Loads all aligned spans across all languages from the align phase, scores
    them together in a single GPU batch for maximum utilization, aggregates
    scores to document level, and writes per-language output.jsonl files
    (same structure consumed by the existing merge step).
    """
    try:
        from segale_eval import run_comet_qe_evaluation, run_metricx_qe_evaluation
    except ImportError as e:
        raise ImportError("segale_eval could not be imported. Ensure the SEGALE package is installed.") from e

    inter_dir = output_dir / "segale_intermediate"
    align_dir = inter_dir / "align"
    per_lang_base = output_dir / "per_lang"

    # Load original records so scores can be patched back in.
    with open(input_file, "rt", encoding="utf-8") as f:
        orig_records = [json.loads(line) for line in f if line.strip()]

    # Determine which languages have completed alignment.
    if align_dir.exists():
        available_langs = sorted(d.name for d in align_dir.iterdir() if d.is_dir() and (d / "align.done").exists())
    else:
        available_langs = []

    if target_languages:
        available_langs = [lang for lang in available_langs if lang in target_languages]

    if not available_langs:
        LOG.warning("score: no completed align languages found in %s", align_dir)
        return

    # Load aligned spans from each language, tracking which span belongs to which lang.
    lang_spans: dict[str, list[dict]] = {}
    for lang in available_langs:
        spans_file = align_dir / lang / "aligned_spans.jsonl"
        if not spans_file.exists():
            LOG.warning("score: aligned_spans.jsonl missing for lang=%s — skipping", lang)
            continue
        spans: list[dict] = []
        with open(spans_file, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    spans.append(json.loads(line))
        if spans:
            lang_spans[lang] = spans
        else:
            LOG.warning("score: empty aligned_spans.jsonl for lang=%s — skipping", lang)

    # Flatten all spans into one list for a single GPU batch call.
    all_spans: list[dict] = []
    all_span_langs: list[str] = []
    for lang, spans in lang_spans.items():
        for span in spans:
            all_spans.append(span)
            all_span_langs.append(lang)

    if not all_spans:
        LOG.warning("score: no aligned spans to score across any language")
        return

    LOG.info("score: scoring %d spans across %d languages", len(all_spans), len(lang_spans))

    # Single GPU batch for COMET-QE and MetricX-QE.
    qe_windows = [(s["src"], s["tgt"]) for s in all_spans]
    comet_qe_scores = run_comet_qe_evaluation(qe_windows)
    metricx_qe_scores = run_metricx_qe_evaluation(qe_windows)

    for idx, span in enumerate(all_spans):
        span["comet-qe"] = comet_qe_scores[idx]
        span["metricx-qe"] = metricx_qe_scores[idx]

    # Compute language fidelity per (doc_id, lang) from original records.
    # In the fan-out format (document_translation.py), only the first record per
    # document carries the full generation; all later records have generation="".
    # Preserve the first non-empty generation to avoid fidelity always collapsing to 1.0.
    # Result: one entry per (doc_id, target_language) pair — e.g. ~9 350 keys for
    # wmt24pp.doc (170 docs × 55 langs) or 55 keys for wmt24pp.long (1 mega-doc × 55 langs).
    orig_by_key: dict[tuple[str, str], dict] = {}
    for r in orig_records:
        if "doc_id" in r:
            key = (r["doc_id"], r.get("target_language", ""))
            if key not in orig_by_key or (r.get("generation") and not orig_by_key[key].get("generation")):
                orig_by_key[key] = r

    doc_lang_fidelity: dict[tuple[str, str], float] = {}
    seen_pairs: set[tuple[str, str]] = set()
    for span, lang in zip(all_spans, all_span_langs):
        key = (span["doc_id"], lang)
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        rec = orig_by_key.get(key)
        if rec:
            ref_text = " ".join(rec.get("reference_sentences", []))
            fidelity = _compute_lang_fidelity(rec.get("generation", ""), ref_text) if ref_text else 1.0
        else:
            fidelity = 1.0
        doc_lang_fidelity[key] = fidelity

    # Aggregate spans to document-level scores and write per-language outputs.
    for lang, spans in lang_spans.items():
        grouped: dict[str, list] = defaultdict(list)
        for span in spans:
            grouped[span["doc_id"]].append(span)

        doc_scores: dict[str, dict] = {}
        for doc_id, doc_spans in grouped.items():
            valid_comet_qe = [s["comet-qe"] for s in doc_spans if s["comet-qe"] >= 0]
            valid_metricx_qe = [s["metricx-qe"] for s in doc_spans if s["metricx-qe"] >= 0]
            fidelity = doc_lang_fidelity.get((doc_id, lang), 1.0)
            doc_scores[doc_id] = {
                "segale_comet_qe": (sum(valid_comet_qe) / len(valid_comet_qe) if valid_comet_qe else 0.0),
                "segale_metricx_qe": (sum(valid_metricx_qe) / len(valid_metricx_qe) if valid_metricx_qe else 0.0),
                "segale_lang_fidelity": fidelity,
                "segale_total_seg": len(doc_spans),
                "segale_misaligned_seg": sum(1 for s in doc_spans if s["comet-qe"] <= 0),
            }

        lang_out_dir = per_lang_base / lang
        lang_out_dir.mkdir(parents=True, exist_ok=True)
        lang_out_file = lang_out_dir / "output.jsonl"

        lang_orig = [r for r in orig_records if r.get("target_language") == lang]
        for record in lang_orig:
            doc_id = record.get("doc_id")
            if doc_id and doc_id in doc_scores:
                record.update(doc_scores[doc_id])

        with open(lang_out_file, "wt", encoding="utf-8") as f:
            for record in lang_orig:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        Path(str(lang_out_file) + ".done").touch()
        LOG.info("score: wrote %s (%d records, %d docs scored)", lang_out_file, len(lang_orig), len(doc_scores))

        if save_spans:
            per_lang_spans = []
            for doc_id, doc_spans in sorted(grouped.items()):
                sorted_spans = sorted(doc_spans, key=lambda s: s.get("seg_id", 0))
                per_lang_spans.append(
                    {
                        "doc_id": doc_id,
                        "lang": lang,
                        "spans": [
                            {
                                "comet_qe": s["comet-qe"],
                                "metricx_qe": s["metricx-qe"],
                                "deleted": not s.get("tgt"),
                                "hallucinated": not s.get("src"),
                            }
                            for s in sorted_spans
                        ],
                    }
                )
            spans_file = lang_out_dir / "spans.jsonl"
            with open(spans_file, "wt", encoding="utf-8") as f:
                for rec in per_lang_spans:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            LOG.info("score: wrote per-span scores to %s (%d docs)", spans_file, len(per_lang_spans))

    LOG.info("Score phase complete for %d languages", len(lang_spans))


_SEGALE_SCORE_KEYS = frozenset(
    {
        "segale_comet",
        "segale_comet_qe",
        "segale_metricx",
        "segale_metricx_qe",
        "segale_bleu",
        "segale_lang_fidelity",
        "segale_total_seg",
        "segale_misaligned_seg",
    }
)


def merge_per_language_outputs(
    original_input: Path,
    langs_dir: Path,
    output_file: Path,
):
    """Merge per-language scored outputs back into the original record order.

    Each per-language judge task writes its scored records to
    ``<langs_dir>/<target_language>/output.jsonl``.  This function reads the
    original input file to recover global record order, builds a score map
    keyed by ``(doc_id, seg_id, target_language)``, patches those scores into
    the original records, and writes the final combined output.
    """
    with open(original_input, "rt", encoding="utf-8") as f:
        original_records = [json.loads(line) for line in f if line.strip()]

    score_map: dict[tuple, dict] = {}
    for lang_file in sorted(langs_dir.glob("*/output.jsonl")):
        with open(lang_file, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if "doc_id" in r:
                    key = (r["doc_id"], r.get("seg_id"), r.get("target_language", ""))
                    scores = {k: v for k, v in r.items() if k in _SEGALE_SCORE_KEYS}
                    if scores:
                        score_map[key] = scores

    if not score_map:
        LOG.warning(
            "merge: no scored records found in %s — upstream judge jobs likely failed or timed out. "
            "Preserving intermediate files for next attempt.",
            langs_dir,
        )
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "wt", encoding="utf-8") as f:
        for record in original_records:
            if "doc_id" in record:
                key = (record["doc_id"], record.get("seg_id"), record.get("target_language", ""))
                if key in score_map:
                    record.update(score_map[key])
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    LOG.info("Merged per-language outputs into %s (%d scored records)", output_file, len(score_map))

    spans_parts = sorted(langs_dir.glob("*/spans.jsonl"))
    if spans_parts:
        combined_spans = output_file.parent / "spans.jsonl"
        with open(combined_spans, "wt", encoding="utf-8") as fout:
            for part in spans_parts:
                with open(part, "rt", encoding="utf-8") as fin:
                    for line in fin:
                        if line.strip():
                            fout.write(line)
        LOG.info("Merged per-language spans into %s (%d languages)", combined_spans, len(spans_parts))

    Path(str(output_file) + ".done").touch()

    # Clean up intermediate files now that the scored output is complete.
    # segale_intermediate (embeddings + aligned spans) lives alongside per_lang/
    # in the judge/ subdirectory — langs_dir.parent is that judge/ dir.
    inter_dir = langs_dir.parent / "segale_intermediate"
    if inter_dir.exists():
        shutil.rmtree(inter_dir)
        LOG.info("Removed intermediate directory %s", inter_dir)


def main():
    parser = argparse.ArgumentParser(description="Run SEGALE document-level MT evaluation (3-phase pipeline)")
    parser.add_argument(
        "--input-file", type=str, help="Path to the input output.jsonl (required for embed and score modes)"
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Path to output directory")
    parser.add_argument(
        "--segmenter",
        type=str,
        choices=["spacy", "ersatz"],
        default="ersatz",
        help="Sentence segmenter for source and MT text (default: ersatz)",
    )
    parser.add_argument(
        "--judge-debug",
        action="store_true",
        default=False,
        help="Debug mode: process only the first 2 documents.",
    )
    parser.add_argument(
        "--target-languages",
        type=str,
        default=None,
        help="Comma-separated target_language values to process (e.g. 'de_Latn,fr_Latn'). "
        "Default: all languages found in the input.",
    )
    parser.add_argument(
        "--merge-input",
        type=str,
        default=None,
        help="Original input file path (merge mode only).",
    )
    parser.add_argument(
        "--merge-langs-dir",
        type=str,
        default=None,
        help="Directory containing per-language subdirs to merge (merge mode only).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file path for merge mode (required for merge).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["embed", "align", "score"],
        default=None,
        help="Phase to run: 'embed' (GPU, segment+embed), "
        "'align' (CPU, vecalign in parallel), or 'score' (GPU, COMET-QE/MetricX-QE).",
    )
    parser.add_argument(
        "--embed-batch-size",
        type=int,
        default=512,
        help="Number of overlap strings per LASER2 encode_sentences() call during embed phase "
        "(default: 512). Increase for more GPU memory, decrease if OOM.",
    )
    parser.add_argument(
        "--save-spans",
        action="store_true",
        default=False,
        help="Write per-span scores to spans.jsonl in each per_lang/<lang>/ dir during "
        "the score phase. The merge phase automatically combines them into spans.jsonl "
        "next to metrics.json when the files are present.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    tgt_langs = set(args.target_languages.split(",")) if args.target_languages else None

    if args.mode == "embed":
        if not args.input_file:
            LOG.error("--input-file is required for --mode embed")
            sys.exit(1)
        _run_embed_phase(
            input_file=Path(args.input_file),
            output_dir=output_dir,
            segmenter=args.segmenter,
            judge_debug=args.judge_debug,
            target_languages=tgt_langs,
            embed_batch_size=args.embed_batch_size,
        )

    elif args.mode == "align":
        _run_align_phase(
            output_dir=output_dir,
            segmenter=args.segmenter,
            judge_debug=args.judge_debug,
            target_languages=tgt_langs,
        )

    elif args.mode == "score":
        if not args.input_file:
            LOG.error("--input-file is required for --mode score")
            sys.exit(1)
        _run_score_phase(
            input_file=Path(args.input_file),
            output_dir=output_dir,
            target_languages=tgt_langs,
            save_spans=args.save_spans,
        )

    elif args.merge_input and args.merge_langs_dir:
        if not args.output_file:
            LOG.error("--output-file is required for merge mode")
            sys.exit(1)
        merge_per_language_outputs(
            original_input=Path(args.merge_input),
            langs_dir=Path(args.merge_langs_dir),
            output_file=Path(args.output_file),
        )

    else:
        LOG.error("Specify --mode {embed,align,score} or --merge-input/--merge-langs-dir.")
        sys.exit(1)

    LOG.info("All files processed.")


if __name__ == "__main__":
    main()
