#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from typing import Dict, Optional, Tuple

SCORE_A_RE = re.compile(r"(?i)score\s*A\s*:\s*([-+]?\d+(?:\.\d+)?)")
SCORE_B_RE = re.compile(r"(?i)score\s*B\s*:\s*([-+]?\d+(?:\.\d+)?)")
# Accept both spellings (Judgment/Judgement), bracketed or unbracketed, anywhere in text
JUDGMENT_RE = re.compile(r"(?i)judg(?:e)?ment\s*:\s*(?:\[\s*)?([ab])(?:\s*\])?")


def parse_tail_scores_and_winner(generation_text: str) -> Tuple[float, float, str]:
    """Parse scoreA, scoreB, and winner label from a model judgment string."""
    # Normalize unicode spaces and remove markdown bold markers to make parsing robust.
    unicode_spaces = "\u00a0\u202f\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u200b"
    text = str(generation_text)
    for ch in unicode_spaces:
        text = text.replace(ch, " ")
    # Remove markdown asterisks used for bold/italics
    text = text.replace("*", "")
    if text == "":
        print("Empty generation text we will choose random winner and zero scores")
        return 0.0, 0.0, random.choice(["A", "B"])

    m_a = SCORE_A_RE.search(text)
    m_b = SCORE_B_RE.search(text)
    m_j = JUDGMENT_RE.search(text)
    if not m_a or not m_b or not m_j:
        print(
            f"Could not find Score A, Score B, and Judgment in generation text. We will choose random winner and zero scores. {text[-200:]}"
        )
        return 0.0, 0.0, random.choice(["A", "B"])

    score_a = float(m_a.group(1))
    score_b = float(m_b.group(1))
    winner_side = m_j.group(1).upper()
    if winner_side not in ("A", "B"):
        raise ValueError("Winner side is not A or B.")
    return score_a, score_b, winner_side


def try_get_numeric(value) -> Optional[float]:
    """Convert a value to float when possible; otherwise return None."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not (isinstance(value, float) and math.isnan(value)):
        return float(value)
    # Try numeric string
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def extract_cluster_base_score(obj: dict, side: str, explicit_key: Optional[str]) -> Optional[float]:
    """Resolve cluster base score from explicit key or known side-specific aliases."""
    if explicit_key:
        return try_get_numeric(obj.get(explicit_key))

    side_upper = side.upper()  # "A" or "B"
    candidates = [
        f"cluster_{side_upper}_score",
        f"cluster_score_{side_upper}",
        f"{side_upper}_cluster_score",
        f"cluster_{side_upper}_rating",
        f"cluster_{side_upper}_elo",
        f"cluster_{side_upper}_points",
        f"{side_upper}_base_score",
    ]

    for key in candidates:
        if key in obj:
            val = try_get_numeric(obj.get(key))
            if val is not None:
                return val

    # Fallback: heuristic search for keys containing both side marker and 'score'
    for key, val in obj.items():
        kl = str(key).lower()
        if "score" in kl and side_upper.lower() in kl:
            num = try_get_numeric(val)
            if num is not None:
                return num
    return None


def extract_cluster_grade(obj: dict, side: str, explicit_key: Optional[str]) -> Optional[str]:
    """Resolve cluster grade from explicit key or known side-specific aliases."""
    if explicit_key:
        val = obj.get(explicit_key)
        return None if val is None else str(val)

    side_upper = side.upper()  # "A" or "B"
    candidates = [
        f"cluster_{side_upper}_grade",
        f"grade_{side_upper}",
        f"{side_upper}_grade",
        f"cluster_{side_upper}_letter",
        f"grade{side_upper}",
    ]

    for key in candidates:
        if key in obj and obj.get(key) is not None:
            return str(obj.get(key))

    for key, val in obj.items():
        kl = str(key).lower()
        if "grade" in kl and side_upper.lower() in kl and val is not None:
            return str(val)
    return None


def main():
    """Aggregate per-cluster tournament stats from match JSONL into a CSV."""
    parser = argparse.ArgumentParser(
        description="Aggregate tournament scores and wins per cluster from JSONL matches."
    )
    parser.add_argument("input_jsonl", help="Path to the JSONL file (each line is a game).")
    parser.add_argument(
        "--output",
        "-o",
        help="Output CSV path. Defaults to input path with .results.csv suffix.",
    )
    parser.add_argument("--key-a", help="Explicit key name in JSON for cluster A base score (optional).")
    parser.add_argument("--key-b", help="Explicit key name in JSON for cluster B base score (optional).")
    parser.add_argument(
        "--grade-key-a",
        help="Explicit key name in JSON for cluster A grade (optional).",
    )
    parser.add_argument(
        "--grade-key-b",
        help="Explicit key name in JSON for cluster B grade (optional).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print progress and warnings to stderr.",
    )

    args = parser.parse_args()

    input_path = os.path.abspath(args.input_jsonl)
    if not os.path.isfile(input_path):
        print(f"Input file not found: {input_path}", file=sys.stderr)
        sys.exit(2)

    output_path = os.path.abspath(args.output) if args.output else os.path.splitext(input_path)[0] + ".results.csv"

    # stats: (problem_id, cluster) -> {
    #   'tournament_score': float,
    #   'wins': int,
    #   'cluster_score': Optional[float],
    #   'grade': Optional[str],
    #   'games': int,
    #   'home': int,   # appearances as cluster_A
    #   'away': int    # appearances as cluster_B
    # }
    stats: Dict[Tuple[str, str], Dict[str, Optional[float]]] = {}

    processed = 0
    with open(input_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            if not raw_line.strip():
                continue
            try:
                obj = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                print(
                    f"Skipping invalid JSON line at record {processed + 1}: {exc}",
                    file=sys.stderr,
                )
                continue

            # Determine problem/game id for this match; do not mix across ids.
            # Prefer 'id', with fallbacks to common alternatives; store as string.
            game_id_val = obj.get(
                "id",
                obj.get("problem_id", obj.get("question_id", obj.get("ioi_id", "unknown"))),
            )
            game_id = str(game_id_val)

            a_name = obj.get("cluster_A")
            b_name = obj.get("cluster_B")
            solution_id_a = obj.get("solution_id_A")
            solution_id_b = obj.get("solution_id_B")
            if solution_id_a is None:
                solution_id_a = 0
            if solution_id_b is None:
                solution_id_b = 0
            if not isinstance(a_name, str) or not isinstance(b_name, str):
                print("Skipping line missing cluster_A/cluster_B names.", file=sys.stderr)
                continue

            generation_text = obj.get("generation", "")
            try:
                score_a, score_b, winner = parse_tail_scores_and_winner(str(generation_text))
            except Exception as exc:
                print(
                    f"ERROR: could not parse generation tail for match {processed + 1}: {exc} ...{generation_text}",
                    file=sys.stderr,
                )
                sys.exit(1)

            # Include solution_id_A/solution_id_B in the aggregation key (default to 0 when absent)
            a_key = (game_id, solution_id_a, a_name)
            b_key = (game_id, solution_id_b, b_name)

            if a_key not in stats:
                stats[a_key] = {
                    "tournament_score": 0.0,
                    "wins": 0,
                    "cluster_score": None,
                    "grade": None,
                    "games": 0,
                    "home": 0,
                    "away": 0,
                }
                a_base = extract_cluster_base_score(obj, "A", args.key_a)
                if a_base is not None:
                    stats[a_key]["cluster_score"] = a_base
                a_grade = extract_cluster_grade(obj, "A", args.grade_key_a)
                if a_grade is not None:
                    stats[a_key]["grade"] = a_grade
            else:
                if stats[a_key].get("cluster_score") is None:
                    a_base = extract_cluster_base_score(obj, "A", args.key_a)
                    if a_base is not None:
                        stats[a_key]["cluster_score"] = a_base
                if stats[a_key].get("grade") is None:
                    a_grade = extract_cluster_grade(obj, "A", args.grade_key_a)
                    if a_grade is not None:
                        stats[a_key]["grade"] = a_grade

            if b_key not in stats:
                stats[b_key] = {
                    "tournament_score": 0.0,
                    "wins": 0,
                    "cluster_score": None,
                    "grade": None,
                    "games": 0,
                    "home": 0,
                    "away": 0,
                }
                b_base = extract_cluster_base_score(obj, "B", args.key_b)
                if b_base is not None:
                    stats[b_key]["cluster_score"] = b_base
                b_grade = extract_cluster_grade(obj, "B", args.grade_key_b)
                if b_grade is not None:
                    stats[b_key]["grade"] = b_grade
            else:
                if stats[b_key].get("cluster_score") is None:
                    b_base = extract_cluster_base_score(obj, "B", args.key_b)
                    if b_base is not None:
                        stats[b_key]["cluster_score"] = b_base
                if stats[b_key].get("grade") is None:
                    b_grade = extract_cluster_grade(obj, "B", args.grade_key_b)
                    if b_grade is not None:
                        stats[b_key]["grade"] = b_grade

            # Update participation counters
            stats[a_key]["games"] = int(stats[a_key]["games"]) + 1
            stats[a_key]["home"] = int(stats[a_key]["home"]) + 1
            stats[b_key]["games"] = int(stats[b_key]["games"]) + 1
            stats[b_key]["away"] = int(stats[b_key]["away"]) + 1

            stats[a_key]["tournament_score"] = float(stats[a_key]["tournament_score"]) + float(score_a)
            stats[b_key]["tournament_score"] = float(stats[b_key]["tournament_score"]) + float(score_b)

            if winner == "A":
                stats[a_key]["wins"] = int(stats[a_key]["wins"]) + 1
            elif winner == "B":
                stats[b_key]["wins"] = int(stats[b_key]["wins"]) + 1

            processed += 1
            if args.verbose and processed % 1000 == 0:
                print(f"Processed {processed} matches...", file=sys.stderr)

    # Sort by wins desc, then tournament_score desc, then name asc (grouped by id, solution_id)
    sorted_rows = sorted(
        (
            (
                key[0],  # id
                key[1],  # solution_id
                key[2],  # cluster name
                stats[key]["cluster_score"],
                float(stats[key]["tournament_score"]),
                int(stats[key]["wins"]),
                stats[key].get("grade"),
            )
            for key in stats.keys()
        ),
        key=lambda t: (t[0], t[1], -t[5], -t[4], t[2]),
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(
            [
                "id",
                "solution_id",
                "cluster",
                "cluster_score",
                "tournament_score",
                "number_of_wins",
                "grade",
                "games",
                "home",
                "away",
            ]
        )
        for (
            id_str,
            solution_id,
            name,
            base_score,
            tour_score,
            wins,
            grade,
        ) in sorted_rows:
            key = (id_str, solution_id, name)
            games = int(stats[key].get("games", 0))
            home = int(stats[key].get("home", 0))
            away = int(stats[key].get("away", 0))
            writer.writerow(
                [
                    id_str,
                    solution_id,
                    name,
                    "" if base_score is None else base_score,
                    tour_score,
                    wins,
                    "" if grade is None else grade,
                    games,
                    home,
                    away,
                ]
            )

    # Report index (1-based, excluding header) of highest numeric cluster_score in the output
    # If none are numeric, print -1.
    try:
        with open(output_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            max_val = None
            max_idx = -1
            row_idx = 0
            for row in reader:
                row_idx += 1
                val = row.get("cluster_score", "")
                try:
                    num = float(val)
                except Exception:
                    num = None
                if num is not None and (max_val is None or num > max_val):
                    max_val = num
                    max_idx = row_idx
        print(output_path)
        print(max_idx if max_val is not None else -1)
    except Exception:
        # Fallback: still print path and -1 if scanning fails
        print(output_path)
        print(-1)


if __name__ == "__main__":
    main()
