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
import json
import os
import random
import re
import sys
from typing import List, Tuple

import tournament_schedule as ts


def derive_output_path(input_file: str, output_dir: str) -> str:
    base = os.path.basename(input_file)
    derived = re.sub(r"_cluster\.jsonl$", "_schedule.jsonl", base)
    if derived == base:
        derived = base.rsplit(".", 1)[0] + "_schedule.jsonl"
    return os.path.join(output_dir, derived)


def build_directed_edges(n: int, edges: List[Tuple[int, int]], k: int, rng: random.Random) -> List[Tuple[int, int]]:
    per_first = [0] * n
    target_first = k // 2
    directed = []
    idx_edges = list(range(len(edges)))
    rng.shuffle(idx_edges)
    for ei in idx_edges:
        u, v = edges[ei]
        if per_first[u] < target_first and per_first[v] < target_first:
            if per_first[u] < per_first[v]:
                directed.append((u, v))
                per_first[u] += 1
            elif per_first[v] < per_first[u]:
                directed.append((v, u))
                per_first[v] += 1
            else:
                if rng.random() < 0.5:
                    directed.append((u, v))
                    per_first[u] += 1
                else:
                    directed.append((v, u))
                    per_first[v] += 1
        elif per_first[u] < target_first:
            directed.append((u, v))
            per_first[u] += 1
        elif per_first[v] < target_first:
            directed.append((v, u))
            per_first[v] += 1
        else:
            if rng.random() < 0.5:
                directed.append((u, v))
            else:
                directed.append((v, u))

    if k % 2 == 1:
        for i in range(len(directed)):
            a, b = directed[i]
            if per_first[a] > target_first and per_first[b] < target_first:
                directed[i] = (b, a)
                per_first[a] -= 1
                per_first[b] += 1
            elif per_first[b] > target_first and per_first[a] < target_first:
                directed[i] = (a, b)
                per_first[b] -= 1
                per_first[a] += 1

    return directed


def build_simple_schedule(n: int, games_per_player: int, rng: random.Random) -> List[Tuple[int, int]]:
    """
    Simple scheduling: each player plays against games_per_player opponents,
    with each matchup played twice (once as A, once as B).
    Each player ends up playing 2 * games_per_player games total.

    When player i plays player j, this counts as one opponent for both players.
    """
    directed = []
    # Track which pairs have been matched (undirected)
    opponent_sets = [set() for _ in range(n)]

    for i in range(n):
        # How many more opponents does player i need?
        current_opponents = len(opponent_sets[i])
        needed = games_per_player - current_opponents

        if needed <= 0:
            continue

        # Get available opponents (not self, not already matched)
        available = [j for j in range(n) if j != i and j not in opponent_sets[i]]

        # Shuffle and pick needed opponents
        rng.shuffle(available)
        selected_opponents = available[:needed]

        # Add games both ways and mark as opponents for both players
        for j in selected_opponents:
            directed.append((i, j))
            directed.append((j, i))
            opponent_sets[i].add(j)
            opponent_sets[j].add(i)

    return directed


def write_schedule_jsonl(
    out_path: str,
    cluster_ids: List[str],
    directed: List[Tuple[int, int]],
    reps: dict,
    problem_meta: dict,
    progress: bool,
) -> int:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    count = 0
    total = len(directed)
    last_shown = -1
    with open(out_path, "w") as f:
        for idx, (a, b) in enumerate(directed):
            cid_a = cluster_ids[a]
            cid_b = cluster_ids[b]
            ra = reps.get(cid_a, {"code": "", "grade": [], "cluster_score": 0.0})
            rb = reps.get(cid_b, {"code": "", "grade": [], "cluster_score": 0.0})
            payload = {
                "cluster_A": cid_a,
                "cluster_B": cid_b,
                "code_A": ra.get("code", ""),
                "code_B": rb.get("code", ""),
                "grade_A": ra.get("grade", []),
                "grade_B": rb.get("grade", []),
                "cluster_score_A": ra.get("cluster_score", 0.0),
                "cluster_score_B": rb.get("cluster_score", 0.0),
            }
            if problem_meta:
                payload.update(problem_meta)
            f.write(json.dumps(payload) + "\n")
            count += 1
            if progress and total > 0:
                pct = int(((idx + 1) * 100) / total)
                if pct != last_shown:
                    last_shown = pct
                    bar_width = 40
                    filled = int(pct * bar_width / 100)
                    bar = "#" * filled + "-" * (bar_width - filled)
                    sys.stderr.write(f"\r[{out_path}] [{bar}] {pct}% ({idx + 1}/{total})")
                    sys.stderr.flush()
    if progress:
        sys.stderr.write("\n")
    return count


def write_intracluster_schedule_jsonl(
    f,
    cluster_id: str,
    codes: List[dict],
    directed: List[Tuple[int, int]],
    problem_meta: dict,
) -> int:
    """Write schedule for intra-cluster games (games between solutions within the same cluster)."""
    count = 0
    for a, b in directed:
        code_a = codes[a]
        code_b = codes[b]
        # Handle both "grade" and "score" fields
        grade_a = code_a.get("grade", code_a.get("score", []))
        grade_b = code_b.get("grade", code_b.get("score", []))
        payload = {
            "cluster_A": cluster_id,
            "cluster_B": cluster_id,
            "solution_id_A": a,
            "solution_id_B": b,
            "code_A": code_a.get("code", ""),
            "code_B": code_b.get("code", ""),
            "grade_A": grade_a,
            "grade_B": grade_b,
            "cluster_score_A": 0.0,
            "cluster_score_B": 0.0,
        }
        if problem_meta:
            payload.update(problem_meta)
        f.write(json.dumps(payload) + "\n")
        count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description="Run tournament schedules across problems and report line counts.")
    parser.add_argument("input_dir", type=str, help="Directory containing <problem>_cluster.jsonl files")
    parser.add_argument("games_per_cluster", type=int, help="Number of games per cluster")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for schedules")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--remove-empty",
        action="store_true",
        help="Remove clusters with empty-output hash",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print per-problem progress bars while writing schedules",
    )
    parser.add_argument(
        "--selection-strategy",
        type=str,
        default="longest",
        choices=["longest", "random", "wins"],
        help="How to pick per-cluster representative code",
    )
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the dataset jsonl file")
    parser.add_argument(
        "--intracluster",
        action="store_true",
        help="Schedule games within clusters (intra-cluster) instead of between clusters (inter-cluster)",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Use simple scheduling: each player plays games_per_cluster opponents twice (once as A, once as B), total 2*games_per_cluster games per player",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    totals = {}

    cluster_files = [
        os.path.join(args.input_dir, f)
        for f in os.listdir(args.input_dir)
        if f.endswith("_cluster.jsonl") and os.path.isfile(os.path.join(args.input_dir, f))
    ]
    if not cluster_files:
        print(f"No *_cluster.jsonl files found in {args.input_dir}")
        return

    def problem_num_from_path(path: str) -> int:
        try:
            return int(os.path.basename(path).split("_", 1)[0])
        except Exception:
            return 10**9

    cluster_files.sort(key=problem_num_from_path)

    for in_file in cluster_files:
        p = ts.extract_problem_number_from_cluster_path(in_file)

        clusters = ts.load_clusters(in_file)
        if args.remove_empty:
            clusters = ts.remove_empty_output_clusters(clusters)

        prob_meta = ts.load_problem_metadata(ts.extract_problem_number_from_cluster_path(in_file), args.dataset_path)

        if args.intracluster:
            # Intra-cluster: schedule games within each cluster
            out_path = derive_output_path(in_file, args.output_dir)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            total_games = 0
            with open(out_path, "w") as f:
                for cluster_id, cluster_data in clusters.items():
                    codes = cluster_data.get("codes", [])
                    if not isinstance(codes, list) or len(codes) < 2:
                        continue

                    n = len(codes)
                    # Generate schedule based on mode
                    if args.simple:
                        directed = build_simple_schedule(n, args.games_per_cluster, rng)
                    else:
                        edges = ts.generate_k_regular_fast(n, args.games_per_cluster, rng)
                        directed = build_directed_edges(n, edges, args.games_per_cluster, rng)

                    cnt = write_intracluster_schedule_jsonl(f, cluster_id, codes, directed, prob_meta)
                    total_games += cnt

            totals[p] = total_games
            mode_str = "simple" if args.simple else "k-regular"
            print(f"{p}: {total_games} (intra-cluster, {mode_str})")
        else:
            # Inter-cluster: schedule games between clusters (original behavior)
            cluster_ids = list(clusters.keys())
            n = len(cluster_ids)
            if n < 2:
                print(f"skip {p}: not enough clusters ({n})")
                continue

            reps = ts.compute_cluster_representatives(clusters, rng, selection_strategy=args.selection_strategy)
            # Generate schedule based on mode
            if args.simple:
                directed = build_simple_schedule(n, args.games_per_cluster, rng)
            else:
                edges = ts.generate_k_regular_fast(n, args.games_per_cluster, rng)
                directed = build_directed_edges(n, edges, args.games_per_cluster, rng)
            out_path = derive_output_path(in_file, args.output_dir)
            cnt = write_schedule_jsonl(out_path, cluster_ids, directed, reps, prob_meta, args.progress)
            totals[p] = cnt
            mode_str = "simple" if args.simple else "k-regular"
            print(f"{p}: {cnt} ({mode_str})")

    # Summary
    done = sorted(totals.items())
    if done:
        print("\nSummary:")
        for p, c in done:
            print(f"{p}: {c}")


if __name__ == "__main__":
    main()
