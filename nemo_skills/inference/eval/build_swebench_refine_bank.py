#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Build a fixed attempt-0 bank for controlled SWE-bench refine evals.

The bank freezes a first-round model patch and the verifier evidence for that patch.
Later controlled refine runs can start from this bank instead of re-running round 0,
so different refine prompts are compared on exactly the same failing patch/log pairs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("rt", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "instance_id" not in row:
                raise ValueError(f"Missing instance_id in {path}:{line_no}")
            rows.append(row)
    return rows


def index_by_instance(rows: list[dict], path: Path) -> dict[str, dict]:
    indexed = {}
    for row in rows:
        instance_id = row["instance_id"]
        if instance_id in indexed:
            raise ValueError(f"Duplicate instance_id={instance_id} in {path}")
        indexed[instance_id] = row
    return indexed


def latest_match(root: Path | None, instance_id: str, filename: str) -> str | None:
    if root is None:
        return None
    matches = [p for p in root.glob(f"**/{instance_id}/{filename}") if p.is_file()]
    if not matches:
        return None
    return str(max(matches, key=lambda p: p.stat().st_mtime))


def latest_match_any(root: Path | None, instance_id: str, filenames: tuple[str, ...]) -> str | None:
    matches = []
    if root is None:
        return None
    for filename in filenames:
        matches.extend(p for p in root.glob(f"**/{instance_id}/{filename}") if p.is_file())
    if not matches:
        return None
    return str(max(matches, key=lambda p: p.stat().st_mtime))


def tail_text(path: str | None, max_chars: int) -> str:
    if not path:
        return ""
    text = Path(path).read_text(errors="ignore")
    if max_chars <= 0:
        return text
    return text[-max_chars:]


def normalize_patch(patch):
    if isinstance(patch, str) and not patch.strip():
        return None
    return patch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-output", required=True, type=Path)
    parser.add_argument("--verified-output", required=True, type=Path)
    parser.add_argument("--eval-output-dir", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify-feedback-chars", type=int, default=8000)
    parser.add_argument("--allow-missing-logs", action="store_true")
    args = parser.parse_args()

    baseline_rows = read_jsonl(args.baseline_output)
    verified_rows = read_jsonl(args.verified_output)
    verified_by_id = index_by_instance(verified_rows, args.verified_output)

    missing_verified = [row["instance_id"] for row in baseline_rows if row["instance_id"] not in verified_by_id]
    if missing_verified:
        shown = ", ".join(missing_verified[:10])
        raise ValueError(f"Verified output is missing {len(missing_verified)} instances, e.g. {shown}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    patch_mismatches = []
    missing_logs = []

    with args.output.open("wt", encoding="utf-8") as fout:
        for base_row in baseline_rows:
            instance_id = base_row["instance_id"]
            verified_row = verified_by_id[instance_id]
            base_outputs = dict(base_row.get("swe-bench-outputs") or {})
            verified_outputs = dict(verified_row.get("swe-bench-outputs") or {})
            model_patch = normalize_patch(base_outputs.get("model_patch"))
            verified_patch = normalize_patch(verified_outputs.get("model_patch"))

            if model_patch != verified_patch:
                patch_mismatches.append(instance_id)

            trajectory = base_outputs or verified_outputs
            trajectory = dict(trajectory)
            trajectory["instance_id"] = instance_id
            trajectory["model_patch"] = model_patch
            trajectory.setdefault("model_name_or_path", verified_outputs.get("model_name_or_path", "fixed_attempt0"))

            report = dict(verified_row.get("swe-bench-metrics") or {})
            report_file = latest_match(args.eval_output_dir, instance_id, "report.json")
            test_output_log = latest_match_any(args.eval_output_dir, instance_id, ("test_output.log", "test_output.txt"))
            verify_feedback = tail_text(test_output_log, args.verify_feedback_chars)

            if report.get("patch_exists") is False:
                counts["no_patch"] += 1
            elif not test_output_log and not report.get("resolved"):
                missing_logs.append(instance_id)
            if report.get("resolved"):
                counts["resolved"] += 1
            if model_patch is not None:
                counts["model_patch_exists"] += 1
            if test_output_log:
                counts["has_test_output"] += 1

            bank_row = {
                "instance_id": instance_id,
                "model_patch": model_patch,
                "trajectory": trajectory,
                "report": report,
                "verify_feedback": verify_feedback,
                "eval_artifacts": {
                    "run_id": "eval-outputs",
                    "report_file": report_file,
                    "test_output_log": test_output_log,
                    "verify_feedback_chars": len(verify_feedback),
                    "eval_error": None,
                },
                "source": {
                    "baseline_output": str(args.baseline_output),
                    "verified_output": str(args.verified_output),
                    "eval_output_dir": str(args.eval_output_dir) if args.eval_output_dir else None,
                },
                "baseline_report": base_row.get("swe-bench-metrics"),
            }
            fout.write(json.dumps(bank_row) + "\n")
            counts["total"] += 1

    if patch_mismatches:
        shown = ", ".join(patch_mismatches[:10])
        raise ValueError(f"Patch mismatch between baseline and verified outputs for {len(patch_mismatches)} rows: {shown}")
    if missing_logs and not args.allow_missing_logs:
        shown = ", ".join(missing_logs[:10])
        raise ValueError(
            f"Missing test_output.log/test_output.txt for {len(missing_logs)} unresolved rows, e.g. {shown}. "
            "Pass --allow-missing-logs if this is expected."
        )

    print(json.dumps({"output": str(args.output), **dict(counts)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
