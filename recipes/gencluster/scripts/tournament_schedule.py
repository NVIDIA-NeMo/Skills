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
from typing import Any, Dict, List, Set, Tuple


def load_clusters(cluster_file: str) -> Dict[str, Any]:
    with open(cluster_file, "r") as f:
        return json.load(f)


def remove_empty_output_clusters(clusters: Dict[str, Any]) -> Dict[str, Any]:
    empty_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    filtered: Dict[str, Any] = {}
    for cid, cdata in clusters.items():
        out = cdata.get("output", [])
        if isinstance(out, tuple):
            out = list(out)
        has_empty = any(o == empty_hash for o in out)
        if not has_empty:
            filtered[cid] = cdata
    return filtered


def compute_cluster_representatives(
    clusters: Dict[str, Any], rng: random.Random, selection_strategy: str = "longest"
) -> Dict[str, Dict[str, Any]]:
    reps: Dict[str, Dict[str, Any]] = {}
    for cid, cdata in clusters.items():
        codes = cdata.get("codes", [])
        best_tokens = -1
        best_code = ""
        best_grade = []
        best_sum_grade = float("-inf")
        # Representative selection based on strategy
        if isinstance(codes, list) and len(codes) > 0:
            if selection_strategy == "random":
                s = rng.choice(codes)
                best_code = s.get("code", "")
                best_grade = s.get("grade", s.get("score", []))
            elif selection_strategy == "wins":
                # Choose solution with largest tournament_wins; tie-break by tokens
                def wins_tokens_tuple(s: Dict[str, Any]) -> Tuple[int, int]:
                    try:
                        w = int(float(s.get("tournament_wins", 0)))
                    except Exception:
                        w = 0
                    try:
                        t = int(float(s.get("tokens", 0)))
                    except Exception:
                        t = 0
                    return (w, t)

                s = max(codes, key=wins_tokens_tuple)
                best_code = s.get("code", "")
                best_grade = s.get("grade", s.get("score", []))
            else:
                # Default "longest": representative by largest tokens
                for s in codes:
                    tokens = s.get("tokens", 0)
                    if isinstance(tokens, (int, float)) and tokens > best_tokens:
                        best_tokens = tokens
                        best_code = s.get("code", "")
                        best_grade = s.get("grade", s.get("score", []))
        # Cluster score = max sum(grade or score)
        for s in codes if isinstance(codes, list) else []:
            # Handle both "grade" and "score" fields
            grade = s.get("grade", s.get("score", []))
            try:
                total = float(sum(float(x) for x in grade)) if isinstance(grade, list) else 0.0
            except Exception:
                total = 0.0
            if total > best_sum_grade:
                best_sum_grade = total
        if best_sum_grade == float("-inf"):
            best_sum_grade = 0.0
        reps[cid] = {
            "code": best_code,
            "grade": best_grade,
            "cluster_score": best_sum_grade,
        }
    return reps


def extract_problem_number_from_cluster_path(cluster_path: str) -> int:
    base = os.path.basename(cluster_path)
    m = re.match(r"^(\d+)_cluster\.jsonl$", base)
    if not m:
        return -1
    try:
        return int(m.group(1))
    except Exception:
        return -1


def load_problem_metadata(problem_number: int, meta_path: str) -> Dict[str, Any]:
    # Read from a single JSONL file with many lines and find the one with matching id
    if problem_number < 0:
        return {}
    if not os.path.isfile(meta_path):
        return {}
    try:
        with open(meta_path, "r") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    data = json.loads(s)
                    if not isinstance(data, dict):
                        continue
                    # Check if this entry matches the problem_number
                    if data.get("id") == problem_number:
                        keys = ["id", "name", "ioi_id", "subtask", "problem"]
                        return {k: data.get(k) for k in keys if k in data}
                except json.JSONDecodeError:
                    continue
        # If no match found, return empty dict
        return {}
    except Exception:
        return {}


def generate_k_regular_simple_graph(num_nodes: int, k: int, rng: random.Random) -> List[Tuple[int, int]]:
    if k >= num_nodes:
        k = num_nodes - 1
    if k < 0:
        k = 0
    # Existence condition for simple k-regular graph: if k is odd, num_nodes must be even
    if (k % 2 == 1) and (num_nodes % 2 == 1):
        k -= 1
    if k < 0:
        raise ValueError("Cannot construct non-negative degree graph with given parameters")

    # Try configuration model with rejection to avoid self-loops and multi-edges
    max_attempts = 2000
    indices = list(range(num_nodes))
    for _ in range(max_attempts):
        stubs = []
        for i in indices:
            stubs.extend([i] * k)
        rng.shuffle(stubs)
        ok = True
        edges: Set[Tuple[int, int]] = set()
        for a, b in zip(stubs[0::2], stubs[1::2]):
            if a == b:
                ok = False
                break
            u, v = (a, b) if a < b else (b, a)
            if (u, v) in edges:
                ok = False
                break
            edges.add((u, v))
        if ok and len(edges) == (num_nodes * k) // 2:
            return sorted(edges)

    # Fallback: randomized greedy with restarts and light backtracking
    max_restarts = 50
    for _ in range(max_restarts):
        target_deg = [k] * num_nodes
        edges: Set[Tuple[int, int]] = set()
        neighbors = [set() for _ in range(num_nodes)]
        progress = True
        while progress:
            progress = False
            candidates = [i for i in range(num_nodes) if target_deg[i] > 0]
            if not candidates:
                break
            rng.shuffle(candidates)
            for u in candidates:
                if target_deg[u] <= 0:
                    continue
                vs = [v for v in candidates if v != u and target_deg[v] > 0 and v not in neighbors[u]]
                rng.shuffle(vs)
                added = False
                for v in vs:
                    a, b = (u, v) if u < v else (v, u)
                    if (a, b) in edges:
                        continue
                    edges.add((a, b))
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    target_deg[u] -= 1
                    target_deg[v] -= 1
                    progress = True
                    added = True
                    break
                if not added and target_deg[u] > 0:
                    # light backtrack: remove one random edge incident to u or a random edge overall
                    if neighbors[u]:
                        v = rng.choice(list(neighbors[u]))
                        a, b = (u, v) if u < v else (v, u)
                        if (a, b) in edges:
                            edges.remove((a, b))
                            neighbors[u].remove(v)
                            neighbors[v].remove(u)
                            target_deg[u] += 1
                            target_deg[v] += 1
                            progress = True
                    elif edges:
                        a, b = rng.choice(list(edges))
                        edges.remove((a, b))
                        neighbors[a].remove(b)
                        neighbors[b].remove(a)
                        target_deg[a] += 1
                        target_deg[b] += 1
                        progress = True
        if all(d == 0 for d in target_deg):
            return sorted(edges)
    raise RuntimeError("Failed to generate a uniform schedule; try a different seed or reduce K")


def generate_k_regular_fast(num_nodes: int, k: int, rng: random.Random) -> List[Tuple[int, int]]:
    """Generate a simple k-regular undirected graph on num_nodes via circulant offsets.
    Conditions: k < num_nodes and k*num_nodes must be even. If not, adjust k downward like legacy.
    """
    print(f"Generating k-regular fast graph with {num_nodes} nodes and {k} edges")
    if k >= num_nodes:
        k = num_nodes - 1
    if k < 0:
        k = 0
    if (k % 2 == 1) and (num_nodes % 2 == 1):
        k -= 1
    if k <= 0:
        return []
    n = num_nodes
    edges: Set[Tuple[int, int]] = set()
    # Randomly permute vertex labels to avoid structured pairings on original indices
    perm = list(range(n))
    rng.shuffle(perm)
    if k % 2 == 0:
        max_off = (n - 1) // 2
        pool = list(range(1, max_off + 1))
        rng.shuffle(pool)
        offsets = pool[: (k // 2)]
    else:
        # n even here
        max_off = (n // 2) - 1
        pool = list(range(1, max_off + 1))
        rng.shuffle(pool)
        offsets = pool[: ((k - 1) // 2)]
        offsets.append(n // 2)
    for i in range(n):
        for d in offsets:
            j = (i + d) % n
            u = perm[i]
            v = perm[j]
            a, b = (u, v) if u < v else (v, u)
            if a != b:
                edges.add((a, b))
    return sorted(edges)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a balanced random tournament schedule (K-regular) for clusters in a file."
    )
    parser.add_argument("cluster_file", type=str, help="Path to <subtask>_cluster.jsonl")
    parser.add_argument("games_per_cluster", type=int, help="Number of games per cluster (K)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional explicit output file path (overrides output-dir)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write the derived <name>_schedule.jsonl",
    )
    parser.add_argument(
        "--remove-empty",
        action="store_true",
        help="Remove clusters that contain the empty-output hash",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print a progress bar while generating schedule",
    )
    parser.add_argument(
        "--selection-strategy",
        type=str,
        default="longest",
        choices=["longest", "random", "wins"],
        help="How to pick per-cluster representative code",
    )
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to the dataset jsonl file")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    clusters = load_clusters(args.cluster_file)
    if args.remove_empty:
        clusters = remove_empty_output_clusters(clusters)
    cluster_ids = list(clusters.keys())
    reps = compute_cluster_representatives(clusters, rng=rng, selection_strategy=args.selection_strategy)
    problem_number = extract_problem_number_from_cluster_path(args.cluster_file)
    problem_meta = load_problem_metadata(problem_number, args.dataset_path)
    n = len(cluster_ids)
    if n < 2:
        raise ValueError("Need at least 2 clusters to schedule games")
    print(f"Number of clusters: {n}")
    print(f"Games per cluster: {args.games_per_cluster}")
    # Fast deterministic generator (circulant offsets) for immediate schedule construction
    edges = generate_k_regular_fast(n, args.games_per_cluster, rng)
    print(f"Number of edges: {len(edges)}")
    # Assign directions so each node appears first ~K/2 times
    per_first = [0] * n
    target_first = args.games_per_cluster // 2
    directed = []
    # Randomize processing order to reduce bias
    idx_edges = list(range(len(edges)))
    rng.shuffle(idx_edges)
    for ei in idx_edges:
        u, v = edges[ei]
        # Prefer node below target_first to be first
        if per_first[u] < target_first and per_first[v] < target_first:
            # choose the one with fewer firsts; tie-break random
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
            # both met target; assign randomly
            if rng.random() < 0.5:
                directed.append((u, v))
            else:
                directed.append((v, u))

    # If K is odd, some imbalance is unavoidable; try to smooth residuals
    if args.games_per_cluster % 2 == 1:
        # compute residual desired = floor(K/2) or ceil? keep around target_first
        # attempt swaps across directed pairs where it reduces imbalance
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

    # Prepare JSONL lines
    json_lines = []
    total_games = len(directed)
    last_shown = -1
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
        print(payload)
        # Attach problem metadata when available (e.g., for 24_cluster.jsonl)
        if problem_meta:
            payload.update(problem_meta)
        json_lines.append(json.dumps(payload))
        if args.progress and total_games > 0:
            # update at most 100 times
            pct = int(((idx + 1) * 100) / total_games)
            if pct != last_shown:
                last_shown = pct
                bar_width = 40
                filled = int(pct * bar_width / 100)
                bar = "#" * filled + "-" * (bar_width - filled)
                sys.stderr.write(f"\rProgress: [{bar}] {pct}% ({idx + 1}/{total_games})")
                sys.stderr.flush()
    if args.progress:
        sys.stderr.write("\n")

    # Derive default output path: <input_basename with _cluster.jsonl replaced by _schedule.jsonl>
    out_path = args.out
    if not out_path:
        base = os.path.basename(args.cluster_file)
        derived = re.sub(r"_cluster\.jsonl$", "_schedule.jsonl", base)
        if derived == base:
            derived = base.rsplit(".", 1)[0] + "_schedule.jsonl"
        if args.output_dir:
            out_path = os.path.join(args.output_dir, derived)
        else:
            out_path = None

    if out_path:
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w") as f:
            f.write("\n".join(json_lines))
    else:
        # Fallback to stdout
        print("\n".join(json_lines))


if __name__ == "__main__":
    main()
