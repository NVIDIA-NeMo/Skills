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
import glob
import json
import os
import re
from typing import Dict, Tuple


def load_clusters(path: str) -> Dict[str, dict]:
    with open(path, "r") as f:
        return json.load(f)


def write_clusters(path: str, clusters: Dict[str, dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(clusters, f)


def read_scores_by_problem(csv_path: str, include_solution: bool = False):
    """
    Read tournament CSV.
    If include_solution is False (default):
      Return mapping: problem_id -> (cluster_id -> (tournament_score, tournament_wins, games)).
    If include_solution is True:
      Return mapping: problem_id -> (cluster_id -> (solution_id -> (tournament_score, tournament_wins, games))).
    """
    if include_solution:
        by_problem: Dict[int, Dict[str, Dict[int, Tuple[float, int, int]]]] = {}
    else:
        by_problem: Dict[int, Dict[str, Tuple[float, int, int]]] = {}
    if not csv_path or not os.path.isfile(csv_path):
        return by_problem
    with open(csv_path, "r", newline="") as f:
        sniffer = csv.Sniffer()
        content = f.read()
        f.seek(0)
        has_header = False
        try:
            has_header = sniffer.has_header(content)
        except Exception:
            pass
        reader = csv.reader(f)
        headers = []
        id_idx = None
        cluster_idx = None
        score_idx = None
        wins_idx = None
        games_idx = None
        if has_header:
            try:
                headers = next(reader)
            except StopIteration:
                return by_problem
            headers_l = [h.strip().lower() for h in headers]

            # Identify column indices
            def find_idx(candidates):
                for k in candidates:
                    if k in headers_l:
                        return headers_l.index(k)
                return None

            id_idx = find_idx(["id", "problem_id", "pid"])
            cluster_idx = find_idx(["cluster", "cluster_id", "name"])
            score_idx = find_idx(["tournament_score", "score", "total_score", "points"])
            wins_idx = find_idx(["number_of_wins", "wins", "num_wins", "total_wins"])
            games_idx = find_idx(["games", "number_of_games", "matches", "played", "num_games"])
            solution_id_idx = find_idx(["solution_id", "solution_id_a", "solution_id_b"])
        # Positional fallback for files produced by compute_tournament_score.py:
        # id, solution_id, cluster, cluster_score, tournament_score, number_of_wins, grade, games, home, away
        for row in reader:
            if not row:
                continue
            try:
                if has_header:
                    pid_raw = row[id_idx] if id_idx is not None and id_idx < len(row) else row[0]
                    cluster_id = (
                        str(row[cluster_idx]).strip()
                        if cluster_idx is not None and cluster_idx < len(row)
                        else str(row[2]).strip()
                    )
                    score_val = (
                        row[score_idx]
                        if score_idx is not None and score_idx < len(row)
                        else (row[4] if len(row) > 4 else "0")
                    )
                    wins_val = (
                        row[wins_idx]
                        if wins_idx is not None and wins_idx < len(row)
                        else (row[5] if len(row) > 5 else "0")
                    )
                    games_val = (
                        row[games_idx]
                        if games_idx is not None and games_idx < len(row)
                        else (row[7] if len(row) > 7 else "0")
                    )
                    solution_id_val = (
                        row[solution_id_idx]
                        if solution_id_idx is not None and solution_id_idx < len(row)
                        else (row[1] if len(row) > 1 else "0")
                    )
                else:
                    pid_raw = row[0]
                    solution_id_val = row[1] if len(row) > 1 else "0"
                    cluster_id = str(row[2]).strip() if len(row) > 2 else ""
                    score_val = row[4] if len(row) > 4 else "0"
                    wins_val = row[5] if len(row) > 5 else "0"
                    games_val = row[7] if len(row) > 7 else (row[4] if len(row) > 4 else "0")
                pid = int(str(pid_raw).strip())
            except Exception:
                continue
            try:
                t_score = float(score_val)
            except Exception:
                t_score = 0.0
            try:
                t_wins = int(float(wins_val))
            except Exception:
                t_wins = 0
            try:
                t_games = int(float(games_val))
            except Exception:
                t_games = 0
            try:
                sol_id = int(float(str(solution_id_val).strip()))
            except Exception:
                sol_id = 0
            if pid not in by_problem:
                by_problem[pid] = {}
            if not cluster_id:
                continue
            if include_solution:
                if cluster_id not in by_problem[pid]:
                    by_problem[pid][cluster_id] = {}
                by_problem[pid][cluster_id][sol_id] = (t_score, t_wins, t_games)
            else:
                by_problem[pid][cluster_id] = (t_score, t_wins, t_games)
    return by_problem


def main():
    parser = argparse.ArgumentParser(
        description="Merge a single tournament CSV (first column is problem id) into per-problem cluster files."
    )
    parser.add_argument(
        "cluster_dir",
        type=str,
        help="Input directory with <problem>_cluster.jsonl files (e.g., cluster_100)",
    )
    parser.add_argument(
        "tournament_csv",
        type=str,
        help="Path to the CSV containing columns: id, cluster, tournament_score, number_of_wins, games",
    )
    parser.add_argument("output_dir", type=str, help="Output directory (e.g., cluster_tournament_100)")
    parser.add_argument(
        "--intra_cluster",
        action="store_true",
        help="If set, write tournament fields into clusters[cluster_id]['codes'][solution_id].",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summary = {}

    # Load scores by problem id
    scores_by_problem = read_scores_by_problem(args.tournament_csv, include_solution=args.intra_cluster)

    # Iterate over all *_cluster.jsonl files in the input directory
    pattern = os.path.join(args.cluster_dir, "*_cluster.jsonl")
    in_files = sorted(glob.glob(pattern))
    if not in_files:
        print(f"No files matched pattern: {pattern}")
    for in_path in in_files:
        base = os.path.basename(in_path)
        m = re.match(r"(\d+)_cluster\.jsonl$", base)
        if not m:
            print(f"skip {base}: filename does not match '<id>_cluster.jsonl'")
            continue
        try:
            p = int(m.group(1))
        except Exception:
            print(f"skip {base}: unable to parse problem id")
            continue
        out_path = os.path.join(args.output_dir, f"{p}_cluster.jsonl")
        scores = scores_by_problem.get(p, {})

        clusters = load_clusters(in_path)
        updated = 0
        for cid, cdata in clusters.items():
            if not args.intra_cluster:
                t_score, t_wins, t_games = scores.get(cid, (0.0, 0, 0))
                try:
                    cdata["tournament_score"] = float(t_score)
                except Exception:
                    cdata["tournament_score"] = 0.0
                try:
                    cdata["tournament_wins"] = int(t_wins)
                except Exception:
                    cdata["tournament_wins"] = 0
                try:
                    cdata["games"] = int(t_games)
                except Exception:
                    cdata["games"] = 0
                updated += 1
            else:
                # In intra-cluster mode, write metrics into per-solution entries under 'codes'
                per_solution = scores.get(cid, {})
                codes = cdata.get("codes")
                if not isinstance(per_solution, dict) or codes is None:
                    continue
                # codes may be a list or a dict; support both
                for sol_id, (t_score, t_wins, t_games) in per_solution.items():
                    try:
                        idx = int(sol_id)
                    except Exception:
                        idx = 0
                    target = None
                    if isinstance(codes, list):
                        if 0 <= idx < len(codes):
                            target = codes[idx]
                    elif isinstance(codes, dict):
                        # use string key to avoid accidental numeric-vs-string mismatches
                        key = str(idx)
                        if key not in codes or not isinstance(codes.get(key), dict):
                            codes[key] = {}
                        target = codes[key]
                    if isinstance(target, dict):
                        try:
                            target["tournament_score"] = float(t_score)
                        except Exception:
                            target["tournament_score"] = 0.0
                        try:
                            target["tournament_wins"] = int(t_wins)
                        except Exception:
                            target["tournament_wins"] = 0
                        try:
                            target["games"] = int(t_games)
                        except Exception:
                            target["games"] = 0
                        updated += 1

        write_clusters(out_path, clusters)
        summary[p] = {"clusters": updated}
        print(f"{p}: wrote {updated} clusters to {out_path}")

    # Optional final summary
    # print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
