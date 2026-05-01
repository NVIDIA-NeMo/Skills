# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Utilities for LibriSpeechMix SOT cpWER evaluation."""

from __future__ import annotations

import itertools
import json
import logging
import re
import string
import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Any

LOG = logging.getLogger(__name__)

SPEAKER_TAG_RE = re.compile(r"\[(s\d+)\]", re.IGNORECASE)
PUNCT_TRANSLATION = str.maketrans({char: " " for char in string.punctuation})


def normalize_no_pnc(text: str) -> str:
    """Lowercase text and remove punctuation for no-PnC cpWER."""
    text = text.lower().translate(PUNCT_TRANSLATION)
    return re.sub(r"\s+", " ", text).strip()


def parse_sot_speaker_streams(text: str, *, default_speaker: str = "s0") -> OrderedDict[str, str]:
    """Parse SOT text into speaker streams keyed by speaker token without brackets."""
    streams: OrderedDict[str, list[str]] = OrderedDict()
    current_speaker: str | None = None
    cursor = 0

    for match in SPEAKER_TAG_RE.finditer(text or ""):
        chunk = text[cursor : match.start()].strip()
        if chunk:
            speaker = current_speaker or default_speaker
            streams.setdefault(speaker, []).append(chunk)
        current_speaker = match.group(1).lower()
        streams.setdefault(current_speaker, [])
        cursor = match.end()

    tail = (text or "")[cursor:].strip()
    if tail:
        speaker = current_speaker or default_speaker
        streams.setdefault(speaker, []).append(tail)

    return OrderedDict((speaker, normalize_no_pnc(" ".join(chunks))) for speaker, chunks in streams.items())


def edit_counts(reference: list[str], hypothesis: list[str]) -> dict[str, int]:
    """Return Levenshtein substitutions, insertions, and deletions."""
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    dp: list[list[tuple[int, int, int, int]]] = [[(0, 0, 0, 0) for _ in range(cols)] for _ in range(rows)]

    for row in range(1, rows):
        prev = dp[row - 1][0]
        dp[row][0] = (prev[0] + 1, prev[1], prev[2], prev[3] + 1)
    for col in range(1, cols):
        prev = dp[0][col - 1]
        dp[0][col] = (prev[0] + 1, prev[1], prev[2] + 1, prev[3])

    for row in range(1, rows):
        for col in range(1, cols):
            if reference[row - 1] == hypothesis[col - 1]:
                hit_prev = dp[row - 1][col - 1]
                hit = (hit_prev[0], hit_prev[1], hit_prev[2], hit_prev[3])
                dp[row][col] = hit
                continue

            sub_prev = dp[row - 1][col - 1]
            ins_prev = dp[row][col - 1]
            del_prev = dp[row - 1][col]
            candidates = [
                (sub_prev[0] + 1, sub_prev[1] + 1, sub_prev[2], sub_prev[3]),
                (ins_prev[0] + 1, ins_prev[1], ins_prev[2] + 1, ins_prev[3]),
                (del_prev[0] + 1, del_prev[1], del_prev[2], del_prev[3] + 1),
            ]
            dp[row][col] = min(candidates, key=lambda item: (item[0], item[1], item[3], item[2]))

    errors, substitutions, insertions, deletions = dp[-1][-1]
    return {
        "errors": errors,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "ref_words": len(reference),
    }


def _counts_for_text(reference: str, hypothesis: str) -> dict[str, int]:
    return edit_counts(reference.split(), hypothesis.split())


def cpwer(reference_text: str, hypothesis_text: str) -> dict[str, Any]:
    """Compute cpWER using minimum speaker permutation matching."""
    ref_streams = parse_sot_speaker_streams(reference_text)
    hyp_streams = parse_sot_speaker_streams(hypothesis_text)

    ref_items = list(ref_streams.items())
    hyp_items = list(hyp_streams.items())
    size = max(len(ref_items), len(hyp_items), 1)
    padded_refs = ref_items + [(f"__empty_ref_{idx}", "") for idx in range(size - len(ref_items))]
    padded_hyps = hyp_items + [(f"__empty_hyp_{idx}", "") for idx in range(size - len(hyp_items))]

    best: dict[str, Any] | None = None
    for permutation in itertools.permutations(range(size)):
        totals = {"errors": 0, "substitutions": 0, "insertions": 0, "deletions": 0, "ref_words": 0}
        assignment = []
        for ref_idx, hyp_idx in enumerate(permutation):
            ref_speaker, ref_words = padded_refs[ref_idx]
            hyp_speaker, hyp_words = padded_hyps[hyp_idx]
            counts = _counts_for_text(ref_words, hyp_words)
            for key in totals:
                totals[key] += counts[key]
            assignment.append({"reference": ref_speaker, "hypothesis": hyp_speaker})

        candidate = {
            **totals,
            "cpwer": totals["errors"] / totals["ref_words"] if totals["ref_words"] else 0.0,
            "assignment": assignment,
            "reference_streams": dict(ref_streams),
            "hypothesis_streams": dict(hyp_streams),
        }
        key = (candidate["errors"], candidate["substitutions"], candidate["deletions"], candidate["insertions"])
        if best is None:
            best = candidate
            best_key = key
        elif key < best_key:
            best = candidate
            best_key = key

    return best or {
        "cpwer": 0.0,
        "errors": 0,
        "substitutions": 0,
        "insertions": 0,
        "deletions": 0,
        "ref_words": 0,
        "assignment": [],
        "reference_streams": {},
        "hypothesis_streams": {},
    }


def sot_to_seglst(record: dict[str, Any], text_key: str) -> list[dict[str, Any]]:
    """Convert a SOT transcript into coarse SegLST entries for MeetEval cpWER."""
    text = record.get(text_key, "") or ""
    streams = parse_sot_speaker_streams(text)
    session_id = record.get("sample_id") or record.get("id") or Path(record.get("audio_filepath", "sample")).stem
    offset = float(record.get("offset", 0.0) or 0.0)
    duration = max(float(record.get("duration", 0.0) or 0.0), 0.01)
    step = duration / max(len(streams), 1)

    entries = []
    for idx, (speaker, words) in enumerate(streams.items()):
        start = offset + idx * step
        end = offset + (idx + 1) * step
        entries.append(
            {
                "session_id": str(session_id),
                "words": words,
                "speaker": speaker,
                "start_time": round(start, 2),
                "end_time": round(end, 2),
            }
        )
    return entries


def write_meeteval_artifacts(records: list[dict[str, Any]], output_file: str | Path) -> dict[str, str]:
    """Write reference and hypothesis SegLST JSON files next to an evaluated output file."""
    output_path = Path(output_file)
    stem = output_path.with_suffix("")
    ref_path = stem.parent / f"{stem.name}_ref.seglst.json"
    hyp_path = stem.parent / f"{stem.name}_hyp.seglst.json"
    avg_path = stem.parent / f"{stem.name}_meeteval_average.json"
    per_path = stem.parent / f"{stem.name}_meeteval_per_reco.json"

    ref_entries: list[dict[str, Any]] = []
    hyp_entries: list[dict[str, Any]] = []
    for record in records:
        ref_entries.extend(sot_to_seglst(record, "text"))
        hyp_record = dict(record)
        hyp_record["pred_text"] = record.get("pred_text") or record.get("generation", "")
        hyp_entries.extend(sot_to_seglst(hyp_record, "pred_text"))

    ref_path.write_text(json.dumps(ref_entries, indent=2) + "\n", encoding="utf-8")
    hyp_path.write_text(json.dumps(hyp_entries, indent=2) + "\n", encoding="utf-8")

    artifacts = {"ref": str(ref_path), "hyp": str(hyp_path)}
    command = [
        "meeteval-wer",
        "cpwer",
        "-r",
        str(ref_path),
        "-h",
        str(hyp_path),
        "--normalizer",
        "chime8",
        "--average-out",
        str(avg_path),
        "--per-reco-out",
        str(per_path),
        "--partial",
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except FileNotFoundError:
        LOG.warning("meeteval-wer is not installed; wrote SegLST artifacts but skipped MeetEval cpWER.")
    except subprocess.CalledProcessError as exc:
        LOG.warning("meeteval-wer cpwer failed: %s", exc.stderr.strip() or exc.stdout.strip())
    else:
        artifacts["average"] = str(avg_path)
        artifacts["per_reco"] = str(per_path)
    return artifacts
