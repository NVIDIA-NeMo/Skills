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

GAUSSIAN_CUBE_TERMS = re.compile(
    r"(Gaussian|\\mathbb\{Z\}\[i\]|Z\[i\]|a\^2\s*\+\s*i\s*b\^2|a\^2\+ib\^2)",
    re.IGNORECASE,
)
CUBE_FACTOR_TERMS = re.compile(
    r"(\([a-z]\s*\+\s*[a-z]i\)\^3|\([a-z]\+i[a-z]\)\^3|unit\s+times\s+a\s+cube|associate\s+to\s+a\s+cube)",
    re.IGNORECASE,
)
SIGNED_REAL_FACTOR_TERMS = re.compile(
    r"([a-z]\^2\s*-\s*3[a-z]\^2|3[a-z]\^2\s*-\s*[a-z]\^2|c\^2-3d\^2|3c\^2-d\^2|u\^2-3v\^2|3u\^2-v\^2)",
    re.IGNORECASE,
)
SIGNED_BRANCH_COVERAGE_TERMS = re.compile(
    r"(both\s+negative|negative\s+branches?|sign\s+branches?|branch\s+table|"
    r"c\^2\s*-\s*3d\^2\s*<\s*0|3c\^2\s*-\s*d\^2\s*<\s*0|"
    r"u\^2\s*-\s*3v\^2\s*<\s*0|3u\^2\s*-\s*v\^2\s*<\s*0)",
    re.IGNORECASE,
)


NEGATIVE_FOURTH_POWER_CONGRUENCE = re.compile(
    r"(a\^4.{0,120}(?:\\equiv|==|=).{0,40}-\s*b\^4|"
    r"a\^\{4\}.{0,120}(?:\\equiv|==|=).{0,40}-\s*b\^\{4\})",
    re.IGNORECASE | re.DOTALL,
)
BAD_NEGATIVE_FOURTH_ROOT = re.compile(
    r"(a\^2.{0,100}(?:\\equiv|==|=).{0,40}(?:\\pm|\+/-).{0,30}b\^2|"
    r"a\^\{2\}.{0,100}(?:\\equiv|==|=).{0,40}(?:\\pm|\+/-).{0,30}b\^\{2\})",
    re.IGNORECASE | re.DOTALL,
)

FALSE_NEGATIVE_BRANCH_CUBE_EXPANSION = re.compile(
    r"(s\^\{12\}\s*\+\s*27s\^\{8\}u\^\{4\}\s*\+\s*243s\^\{4\}u\^\{8\}\s*\+\s*729u\^\{12\}.{0,240}"
    r"\(s\^\{4\}\s*\+\s*3u\^\{4\}\)\^\{3\})",
    re.IGNORECASE | re.DOTALL,
)

GAME_BAY_TERMS = re.compile(r"\b(bay|concave|L[- ]shape|missing corner|2x2|2\\times\\s*2)\b", re.IGNORECASE)
GAME_SHAYAN_ALWAYS_K1_TERMS = re.compile(
    r"(Shayan.{0,120}(always|will always|can always|must)\s+(play|choose).{0,80}k\s*=\s*1|"
    r"Shayan'?s\s+moves?.{0,120}(all|always).{0,80}k\s*=\s*1)",
    re.IGNORECASE | re.DOTALL,
)
GAME_BAY_FILL_COVERAGE_TERMS = re.compile(
    r"(fill(s|ing)?\s+the\s+(bay|missing\s+corner)|bay[- ]filling|Shayan.{0,160}(fill|fills|filled|filling).{0,80}(bay|missing\s+corner))",
    re.IGNORECASE | re.DOTALL,
)


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


def gaussian_cube_sign_branch_reasons(proof: str, problem_idx: str | None = None) -> list[str]:
    """Detect Gaussian-cube arguments that omit signed real-factor branches."""

    if problem_idx != "proofbench133_109":
        return []
    if not (GAUSSIAN_CUBE_TERMS.search(proof) and CUBE_FACTOR_TERMS.search(proof)):
        return []
    if not SIGNED_REAL_FACTOR_TERMS.search(proof):
        return []
    if SIGNED_BRANCH_COVERAGE_TERMS.search(proof):
        return []
    return [
        "uses a Gaussian cube factorization and square real-factor equations without an explicit sign/negative-branch analysis"
    ]


def negative_fourth_power_root_reasons(proof: str, problem_idx: str | None = None) -> list[str]:
    if problem_idx != "proofbench133_109":
        return []
    if NEGATIVE_FOURTH_POWER_CONGRUENCE.search(proof) and BAD_NEGATIVE_FOURTH_ROOT.search(proof):
        return ["derives a^2 == +/- b^2 from a negative fourth-power congruence such as a^4 == -b^4 modulo p"]
    return []


def false_negative_branch_cube_expansion_reasons(proof: str, problem_idx: str | None = None) -> list[str]:
    if problem_idx != "proofbench133_109":
        return []
    if FALSE_NEGATIVE_BRANCH_CUBE_EXPANSION.search(proof):
        return [
            "uses the false expansion s^12+27s^8u^4+243s^4u^8+729u^12 = (s^4+3u^4)^3 in a Gaussian negative-branch argument"
        ]
    return []


def game_bay_response_reasons(proof: str, problem_idx: str | None = None) -> list[str]:
    """Detect obvious k=1/bay-strategy false positives for the grid game."""

    if problem_idx != "proofbench133_030":
        return []
    has_always_k1 = bool(GAME_SHAYAN_ALWAYS_K1_TERMS.search(proof))
    if has_always_k1:
        return [
            "bases the grid-game strategy on Shayan can/should always play k=1; this recurring family needs adversarial bay-filling verification before promotion"
        ]
    return []


def annotate_row(row: dict[str, Any]) -> dict[str, Any]:
    proof = str(row.get("proof") or row.get("generation") or row.get("candidate_proof") or "")
    problem_idx = row.get("problem_idx")
    reasons = []
    reasons.extend(pitot_false_converse_reasons(proof))
    reasons.extend(gaussian_cube_sign_branch_reasons(proof, problem_idx=problem_idx))
    reasons.extend(negative_fourth_power_root_reasons(proof, problem_idx=problem_idx))
    reasons.extend(false_negative_branch_cube_expansion_reasons(proof, problem_idx=problem_idx))
    reasons.extend(game_bay_response_reasons(proof, problem_idx=problem_idx))
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
