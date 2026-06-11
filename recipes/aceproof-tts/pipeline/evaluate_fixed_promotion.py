#!/usr/bin/env python3
"""Evaluate fixed-harness promotion from verifier sidecars.

This script intentionally does not rank or reinterpret proof content. It applies
a strict reproducible gate:

1. candidate must pass deterministic static promotion filters;
2. each required verifier family must have enough completed scores;
3. every required verifier score must be exactly 1.0.
"""

from __future__ import annotations

import argparse
import glob
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

STATIC_FILTER_PATH = Path(__file__).with_name("run_static_promotion_filters.py")


def _load_static_filter():
    spec = importlib.util.spec_from_file_location("static_promotion_filters", STATIC_FILTER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _iter_jsonl_paths(path_or_glob: str):
    p = Path(path_or_glob)
    if p.is_dir():
        paths = sorted(str(x) for x in p.glob("output*.jsonl*") if not str(x).endswith(".done"))
    elif p.exists() and not str(p).endswith(".done"):
        paths = [str(p)]
    else:
        paths = sorted(glob.glob(path_or_glob))
    for path in paths:
        if path.endswith(".done"):
            continue
        if Path(path).is_dir():
            continue
        yield path


def _read_jsonl(path: str):
    with open(path, encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            yield json.loads(line)


def _parse_sidecar_specs(specs: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"sidecar spec must be NAME:PATH_OR_GLOB, got {spec!r}")
        name, path = spec.split(":", 1)
        if not name:
            raise ValueError(f"empty sidecar name in {spec!r}")
        out[name].append(path)
    return dict(out)


def _load_scores(sidecars: dict[str, list[str]]) -> dict[str, dict[str, list[dict[str, Any]]]]:
    scores: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for family, patterns in sidecars.items():
        for pattern in patterns:
            for path in _iter_jsonl_paths(pattern):
                for row in _read_jsonl(path):
                    uid = row.get("candidate_uid")
                    if not uid:
                        continue
                    scores[uid][family].append(row)
    return scores


def _score_value(row: dict[str, Any]) -> float | None:
    value = row.get("verification_score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _evaluate_candidate(
    row: dict[str, Any],
    families: list[str],
    rows_by_family: dict[str, list[dict[str, Any]]],
    min_scores_per_family: int,
    require_complete: bool,
) -> tuple[bool, list[str], dict[str, Any]]:
    reasons: list[str] = []
    summary: dict[str, Any] = {}

    for family in families:
        verifier_rows = rows_by_family.get(family, [])
        values = [_score_value(x) for x in verifier_rows]
        complete_rows = [
            x
            for x in verifier_rows
            if (not require_complete or x.get("verification_complete"))
            and x.get("valid", True)
            and _score_value(x) is not None
        ]
        complete_values = [_score_value(x) for x in complete_rows]
        summary[family] = {
            "num_rows": len(verifier_rows),
            "num_complete_rows": len(complete_rows),
            "score_counts": dict(Counter(values)),
            "complete_score_counts": dict(Counter(complete_values)),
        }
        if len(complete_values) < min_scores_per_family:
            reasons.append(
                f"{family}: only {len(complete_values)} completed valid scores, need {min_scores_per_family}"
            )
            continue
        bad = [x for x in complete_values if x != 1.0]
        if bad:
            reasons.append(f"{family}: non-1.0 scores present {dict(Counter(bad))}")

    return not reasons, reasons, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--sidecar", action="append", required=True, help="NAME:PATH_OR_GLOB")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min_scores_per_family", type=int, default=8)
    parser.add_argument("--require_complete", action="store_true", default=True)
    args = parser.parse_args()

    static_filter = _load_static_filter()
    sidecars = _parse_sidecar_specs(args.sidecar)
    families = list(sidecars)
    sidecar_scores = _load_scores(sidecars)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    promoted = 0
    with open(args.candidates, encoding="utf-8") as fin, output.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            candidate = json.loads(line)
            uid = candidate.get("candidate_uid")
            annotated = static_filter.annotate_row(candidate)
            reasons: list[str] = []
            if annotated.get("static_promotion_reject"):
                reasons.extend(f"static: {reason}" for reason in annotated.get("static_promotion_reject_reasons", []))
            pass_sidecars, sidecar_reasons, sidecar_summary = _evaluate_candidate(
                annotated,
                families=families,
                rows_by_family=sidecar_scores.get(uid, {}),
                min_scores_per_family=args.min_scores_per_family,
                require_complete=args.require_complete,
            )
            reasons.extend(sidecar_reasons)
            out = dict(annotated)
            out["fixed_promotion_required_families"] = families
            out["fixed_promotion_sidecar_summary"] = sidecar_summary
            out["fixed_promotion_pass"] = (not reasons) and pass_sidecars
            out["fixed_promotion_reject_reasons"] = reasons
            promoted += int(out["fixed_promotion_pass"])
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"wrote {output}")
    print(f"candidates={total} promoted={promoted}")


if __name__ == "__main__":
    main()
