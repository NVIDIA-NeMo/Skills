#!/usr/bin/env python3
"""Build model-generated attempt-memory inputs from prior candidate pools."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _iter_jsonl(patterns: list[str]):
    for pattern in patterns:
        for file_path in sorted(glob.glob(pattern)):
            with open(file_path, encoding="utf-8") as fin:
                for line in fin:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        continue
                    row["_source_file"] = file_path
                    yield row


def _proof_key(proof: str) -> str:
    return hashlib.sha1(proof.encode("utf-8")).hexdigest()[:16]


def _candidate_uid(row: dict[str, Any], proof: str) -> str:
    if row.get("candidate_uid"):
        return str(row["candidate_uid"])
    problem_idx = row.get("problem_idx", "unknown")
    seed = row.get("generation_seed", row.get("row_id", "noseed"))
    source = row.get("_source_file") or row.get("candidate_source_file") or "unknown"
    parts = Path(source).parts
    arm = "/".join(parts[-5:-3]) if len(parts) >= 5 else "unknown"
    return f"{problem_idx}:{seed}:{arm}:{_proof_key(proof)}"


def _shorten(text: str, limit: int) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "\n...[truncated]"


def _load_base(path: str, targets: set[str] | None) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            problem_idx = row.get("problem_idx")
            if targets and problem_idx not in targets:
                continue
            rows[problem_idx] = row
    return rows


def _collect_verifier_comments(args) -> dict[str, list[dict[str, Any]]]:
    comments: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _iter_jsonl(args.verifier_glob):
        uid = row.get("candidate_uid")
        if not uid:
            continue
        rating = row.get("rating_text") or row.get("generation") or ""
        comments[uid].append(
            {
                "score": row.get("verification_score"),
                "file": row.get("_source_file"),
                "text": _shorten(rating, args.max_verifier_chars),
            }
        )
    return comments


def _collect_candidates(args, base_rows: dict[str, dict[str, Any]], comments: dict[str, list[dict[str, Any]]]):
    dedup: dict[tuple[str, str], dict[str, Any]] = {}
    skipped = 0
    for row in _iter_jsonl(args.candidate_glob):
        problem_idx = row.get("problem_idx")
        if problem_idx not in base_rows:
            continue
        proof = str(row.get("proof") or "").strip()
        if not proof or proof == "UNFINISHED PROOF GENERATION" or not row.get("valid", True):
            skipped += 1
            continue
        key = (problem_idx, proof)
        if key in dedup:
            continue
        uid = _candidate_uid(row, proof)
        dedup[key] = {
            "problem_idx": problem_idx,
            "candidate_uid": uid,
            "generation_seed": row.get("generation_seed"),
            "row_id": row.get("row_id"),
            "self_eval_score": row.get("self_eval_score"),
            "source_file": row.get("_source_file"),
            "proof": proof,
            "verifier_comments": comments.get(uid, []),
        }

    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dedup.values():
        by_problem[row["problem_idx"]].append(row)

    for rows in by_problem.values():
        rows.sort(
            key=lambda r: (
                len(r["verifier_comments"]),
                float(r.get("self_eval_score") or 0),
                len(str(r.get("proof") or "")),
            ),
            reverse=True,
        )
    return by_problem, skipped, len(dedup)


def build(args) -> None:
    targets = set(args.target_problem or []) or None
    base_rows = _load_base(args.base_input, targets)
    comments = _collect_verifier_comments(args)
    by_problem, skipped, dedup_count = _collect_candidates(args, base_rows, comments)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as fout:
        for problem_idx, base in base_rows.items():
            question = base.get("question") or base.get("problem") or base.get("original_problem") or ""
            candidates = by_problem.get(problem_idx, [])[: args.max_candidates_per_problem]
            parts = [
                "## Original Problem",
                question,
                "",
                "## Previous Model Attempt Bundle",
                (
                    "The following candidates and verifier comments were generated by model/harness runs. "
                    "They are not trusted facts and may be wrong."
                ),
            ]
            if not candidates:
                parts.append("\nNo prior completed candidates were found for this problem in the selected pools.")
            for idx, cand in enumerate(candidates, 1):
                parts.extend(
                    [
                        "",
                        f"### Candidate {idx}",
                        f"candidate_uid: {cand['candidate_uid']}",
                        f"generation_seed: {cand.get('generation_seed')}",
                        f"row_id: {cand.get('row_id')}",
                        f"self_eval_score: {cand.get('self_eval_score')}",
                        f"source_file: {cand.get('source_file')}",
                        "",
                        "Proof text:",
                        _shorten(cand["proof"], args.max_candidate_chars),
                    ]
                )
                verifier_comments = cand.get("verifier_comments") or []
                if verifier_comments:
                    parts.append("\nVerifier comments for this candidate:")
                    for cidx, comment in enumerate(verifier_comments[: args.max_comments_per_candidate], 1):
                        parts.extend(
                            [
                                f"- comment {cidx}; score={comment.get('score')}; file={comment.get('file')}",
                                _shorten(comment.get("text") or "", args.max_verifier_chars),
                            ]
                        )
            out_row = dict(base)
            out_row["source_name"] = args.source_name
            out_row["question"] = "\n".join(parts)
            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")

    print(f"base_problems={len(base_rows)}")
    print(f"dedup_candidates={dedup_count}")
    print(f"skipped_invalid={skipped}")
    for problem_idx in base_rows:
        print(f"{problem_idx}: {len(by_problem.get(problem_idx, []))} candidates")
    print(f"wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_input", required=True)
    parser.add_argument("--candidate_glob", action="append", required=True)
    parser.add_argument("--verifier_glob", action="append", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--target_problem", action="append")
    parser.add_argument("--source_name", default="repeat10_attempt_memory_synthesis")
    parser.add_argument("--max_candidates_per_problem", type=int, default=12)
    parser.add_argument("--max_comments_per_candidate", type=int, default=3)
    parser.add_argument("--max_candidate_chars", type=int, default=12000)
    parser.add_argument("--max_verifier_chars", type=int, default=2500)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
