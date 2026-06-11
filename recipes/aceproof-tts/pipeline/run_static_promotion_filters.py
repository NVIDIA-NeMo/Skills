#!/usr/bin/env python3
"""Apply deterministic promotion-sidecar filters to proof candidates.

The filters here are calibration helpers for fixed-harness promotion gates.
They do not modify verifier scores; they add static rejection metadata that can
be combined with normal verifier votes when selecting final candidates.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

TANGENTIAL_TERMS = re.compile(
    r"\b(tangential|circumscriptible|circumscribed|incircle|inscribed circle)\b",
    re.IGNORECASE,
)
SIDE_SUM_TERMS = re.compile(
    r"(\ba\s*\+\s*c\s*=\s*b\s*\+\s*d\b|"
    r"\bAB\s*\+\s*CD\s*=\s*BC\s*\+\s*DA\b|"
    r"\bopposite sides?\b.{0,80}\b(equal|same)\b.{0,80}\b(sum|sums)\b|"
    r"\b(sum|sums)\b.{0,80}\bopposite sides?\b.{0,80}\b(equal|same)\b)",
    re.IGNORECASE | re.DOTALL,
)
IFF_OR_SUFFICIENCY_TERMS = re.compile(
    r"(\bif and only if\b|\biff\b|\bexactly when\b|\bequivalent to\b|"
    r"\bnecessary and sufficient\b|\bconverse\b|\bsufficient\b|\bsuffices\b|"
    r"\btherefore\b.{0,120}\b(tangential|circumscriptible|incircle|inscribed circle)\b|"
    r"\bhence\b.{0,120}\b(tangential|circumscriptible|incircle|inscribed circle)\b)",
    re.IGNORECASE | re.DOTALL,
)
PITOT_TERMS = re.compile(r"\bPitot'?s?\b", re.IGNORECASE)


def pitot_false_converse_reasons(proof: str) -> list[str]:
    """Detect use of Pitot side-sum equality as sufficient for tangency."""

    reasons: list[str] = []
    if not TANGENTIAL_TERMS.search(proof):
        return reasons

    has_side_sum = bool(SIDE_SUM_TERMS.search(proof))
    has_sufficiency_language = bool(IFF_OR_SUFFICIENCY_TERMS.search(proof))
    mentions_pitot = bool(PITOT_TERMS.search(proof))

    if mentions_pitot and has_sufficiency_language:
        reasons.append("mentions Pitot with iff/converse/sufficiency language near a tangency or incircle claim")
    if has_side_sum and has_sufficiency_language:
        reasons.append(
            "uses opposite-side-sum equality with iff/converse/sufficiency language for a tangency or incircle claim"
        )
    return reasons


def annotate_row(row: dict[str, Any]) -> dict[str, Any]:
    proof = str(row.get("proof") or row.get("generation") or "")
    reasons = pitot_false_converse_reasons(proof)
    out = dict(row)
    out["static_promotion_reject"] = bool(reasons)
    out["static_promotion_reject_reasons"] = reasons
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    rejected = 0
    with input_path.open(encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            out = annotate_row(json.loads(line))
            rejected += int(out["static_promotion_reject"])
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"wrote {output_path}")
    print(f"rows={total} static_rejected={rejected}")


if __name__ == "__main__":
    main()
