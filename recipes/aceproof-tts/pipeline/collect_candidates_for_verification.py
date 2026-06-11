#!/usr/bin/env python3
"""Collect proof-generation outputs into verifier-sidecar inputs."""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

STATIC_FILTER_PATH = Path(__file__).with_name("run_static_promotion_filters.py")


def _load_static_filter():
    spec = importlib.util.spec_from_file_location("static_promotion_filters", STATIC_FILTER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _iter_rows(patterns: list[str]):
    for pattern in patterns:
        for file_path in sorted(glob.glob(pattern)):
            with open(file_path, encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    row["candidate_source_file"] = file_path
                    yield row


def _proof_key(proof: str) -> str:
    return hashlib.sha1(proof.encode("utf-8")).hexdigest()[:16]


def _candidate_uid(row: dict[str, Any], proof: str) -> str:
    problem_idx = row.get("problem_idx", "unknown")
    seed = row.get("generation_seed", row.get("row_id", "noseed"))
    source = row.get("candidate_source_file", "unknown")
    arm_parts = Path(source).parts
    arm = "/".join(arm_parts[-5:-3]) if len(arm_parts) >= 5 else "unknown"
    return f"{problem_idx}:{seed}:{arm}:{_proof_key(proof)}"


def collect(args):
    static_filter = _load_static_filter() if args.exclude_static_reject else None
    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    rejected = 0
    total = 0
    skipped_invalid = 0

    for row in _iter_rows(args.input_glob):
        total += 1
        proof = str(row.get("proof") or "").strip()
        if not proof or proof == "UNFINISHED PROOF GENERATION" or not row.get("valid", True):
            skipped_invalid += 1
            continue
        if static_filter is not None:
            annotated = static_filter.annotate_row(row)
            if annotated.get("static_promotion_reject"):
                rejected += 1
                continue
        problem_idx = row.get("problem_idx", "unknown")
        key = (problem_idx, proof)
        if key in dedup:
            continue
        question = row.get("original_problem") or row.get("question") or row.get("problem") or ""
        candidate = dict(row)
        candidate["question"] = question
        candidate["proof"] = proof
        candidate["candidate_uid"] = _candidate_uid(row, proof)
        candidate["candidate_proof_key"] = _proof_key(proof)
        dedup[key] = candidate

    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dedup.values():
        by_problem[row.get("problem_idx", "unknown")].append(row)

    candidates: list[dict[str, Any]] = []
    for problem_idx in sorted(by_problem):
        rows = by_problem[problem_idx]
        rows.sort(
            key=lambda r: (
                float(r.get("self_eval_score") or 0),
                len(str(r.get("proof") or "")),
                str(r.get("candidate_uid") or ""),
            ),
            reverse=True,
        )
        if args.max_per_problem > 0:
            rows = rows[: args.max_per_problem]
        candidates.extend(rows)

    out_candidates = Path(args.output_candidates)
    out_candidates.parent.mkdir(parents=True, exist_ok=True)
    with out_candidates.open("w", encoding="utf-8") as fout:
        for row in candidates:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    out_verify = Path(args.output_verify)
    out_verify.parent.mkdir(parents=True, exist_ok=True)
    with out_verify.open("w", encoding="utf-8") as fout:
        pos = 0
        for row in candidates:
            for verification_seed in range(args.verification_repeats):
                verify_row = dict(row)
                verify_row["verification_seed"] = verification_seed
                verify_row["_async_position"] = pos
                pos += 1
                fout.write(json.dumps(verify_row, ensure_ascii=False) + "\n")

    print(f"input_rows={total}")
    print(f"skipped_invalid={skipped_invalid}")
    print(f"static_rejected={rejected}")
    print(f"dedup_candidates={len(dedup)}")
    print(f"written_candidates={len(candidates)} -> {out_candidates}")
    print(f"written_verify_rows={len(candidates) * args.verification_repeats} -> {out_verify}")
    for problem_idx in sorted(by_problem):
        print(
            f"{problem_idx}: {min(len(by_problem[problem_idx]), args.max_per_problem or len(by_problem[problem_idx]))}/{len(by_problem[problem_idx])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_glob", action="append", required=True)
    parser.add_argument("--output_candidates", required=True)
    parser.add_argument("--output_verify", required=True)
    parser.add_argument("--verification_repeats", type=int, default=8)
    parser.add_argument("--max_per_problem", type=int, default=128)
    parser.add_argument("--exclude_static_reject", action="store_true")
    collect(parser.parse_args())


if __name__ == "__main__":
    main()
