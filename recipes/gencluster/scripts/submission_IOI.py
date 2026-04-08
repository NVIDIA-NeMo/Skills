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

import json
import random
import sys
from itertools import cycle

import numpy as np

max_score_24 = [
    6,
    13,
    17,
    11,
    20,
    15,
    18,
    -1,
    10,
    90,
    -1,
    10,
    13,
    18,
    7,
    11,
    22,
    19,
    -1,
    3,
    15,
    10,
    16,
    14,
    42,
    -1,
    5,
    7,
    7,
    10,
    8,
    22,
    19,
    22,
    -1,
    3,
    7,
    33,
    21,
    36,
]
max_score_25 = [
    4,
    3,
    14,
    18,
    28,
    33,
    8,
    6,
    10,
    11,
    16,
    19,
    5,
    5,
    5,
    5,
    5,
    5,
    5,
    10,
    7,
    8,
    14,
    56,
    5,
    7,
    12,
    15,
    27,
    16,
    18,
    30,
    70,
    10,
    14,
    13,
    21,
    25,
    17,
]


def get_max_score_for_subtask(subtask_number, dataset="ioi24"):
    """Returns the maximum possible score for a given subtask number."""
    max_score = max_score_24 if dataset == "ioi24" else max_score_25
    if 1 <= subtask_number <= len(max_score):
        return max_score[subtask_number - 1]  # Convert to 0-indexed
    return -1


def get_grade_slice_for_problem(problem_name, dataset="ioi24"):
    """Returns the slice of grade array to use for scoring a specific problem group."""
    if dataset == "ioi25":
        if problem_name == "7-12":
            return slice(0, 6)  # First 6 scores (positions 0-5)
        elif problem_name == "13-18":
            return slice(6, 12)  # Second 6 scores (positions 6-11)
    # For all other cases, use all scores
    return slice(None)


def load_cluster_data(filepath):
    """Loads cluster data from a jsonl file."""
    with open(filepath, "r") as f:
        return json.load(f)


def apply_blind_cluster_filtering(clusters, strategy="balanced"):
    """Filter and reweight clusters using only pre-submission information (no grades). Supports only 'balanced'."""
    if strategy != "balanced":
        raise ValueError(f"Unsupported blind filtering strategy: {strategy}")

    # Analyze cluster characteristics using only available pre-submission data
    cluster_stats = {}
    for cluster_name, cluster_info in clusters.items():
        codes = cluster_info.get("codes", [])
        if not codes:
            continue

        tokens = [solution.get("tokens", 0) for solution in codes]
        cluster_stats[cluster_name] = {
            "size": len(codes),
            "avg_tokens": np.mean(tokens) if tokens else 0,
            "data": cluster_info,
        }

    # Balanced weighting between cluster size and token complexity
    weights = {}
    for name, stats in cluster_stats.items():
        size_score = min(stats["size"] / 50.0, 1.5)
        token_score = stats["avg_tokens"] / 100000.0
        weights[name] = 0.5 * size_score + 0.5 * token_score

    sorted_clusters = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    final_clusters = [(name, cluster_stats[name]["data"]) for name, _ in sorted_clusters]

    return final_clusters


def get_solution_iterator(clusters):
    """Creates an iterator that cycles through clusters first, then solutions within clusters."""
    cluster_names = list(clusters.keys())
    cluster_iterators = {name: cycle(enumerate(clusters[name]["codes"])) for name in cluster_names}
    cluster_cycle = cycle(cluster_names)

    def next_solution():
        """Return the next solution tuple from the round-robin cluster cycle."""
        current_cluster = next(cluster_cycle)
        solution_index, solution = next(cluster_iterators[current_cluster])
        return current_cluster, solution_index, solution

    class SolutionIterator:
        """Iterator wrapper that yields round-robin solutions across clusters."""
        def __iter__(self):
            """Return this iterator instance."""
            return self

        def __next__(self):
            """Return the next solution produced by `next_solution`."""
            return next_solution()

    return SolutionIterator()


def run_submission(
    strategy="wins",
    max_clusters=5,
    dataset="ioi24",
    data_subdir=None,
    in_cluster_strategy: str = "longest",
):
    """Main function to run the submission process.

    Args:
        strategy (str): Inter-cluster sampling strategy to use:
            - "wins":   Sort clusters by merged tournament_wins (higher first)
            - "score":  Sort clusters by merged tournament_score (higher first)
            - "size":   Sort clusters by cluster size (more codes first)
            - "random": Shuffle clusters randomly
        max_clusters (int): Maximum number of top clusters to use per subtask (default: 5, use -1 for all).
        dataset (str): Dataset to use: "ioi24" or "ioi25".
    """
    groups_24 = {
        "1-7": list(range(1, 8)),
        "9-10": [9, 10],
        "12-18": list(range(12, 19)),
        "20-25": list(range(20, 26)),
        "27-34": list(range(27, 35)),
        "36-40": list(range(36, 41)),
    }
    groups_25 = {
        "1-6": list(range(1, 7)),
        "7-12": list(range(7, 13)),  # First part of problem 2 (look at first 6 scores)
        "13-18": list(range(13, 19)),  # Second part of problem 2 (look at second 6 scores)
        "19-24": list(range(19, 25)),
        "25-31": list(range(25, 32)),
        "32-33": list(range(32, 34)),
        "34-39": list(range(34, 40)),
    }

    # Select the appropriate dataset configuration
    if dataset == "ioi24":
        groups = groups_24
        default_dir = "ioi_24/0"
    elif dataset == "ioi25":
        groups = groups_25
        default_dir = "ioi_25/1"
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'ioi24' or 'ioi25'")

    # Allow overriding the data directory (folder) for experiments across folders
    if data_subdir:
        data_dir = data_subdir
    else:
        data_dir = default_dir

    print(f"Using dataset: {dataset.upper()} with {len(groups)} problem groups")
    if dataset == "ioi25":
        print("  Note: Problem 2 split into '7-12' (first 6 scores) and '13-18' (second 6 scores)")
    print(f"  Intra-cluster (within-cluster) strategy: {in_cluster_strategy}")

    # Initialize a dictionary to store the max score for each subtask of each problem
    max_scores = {problem_num: [0.0] * len(subtasks) for problem_num, subtasks in groups.items()}

    # Track which subtasks have reached their maximum possible score
    completed_subtasks = {problem_num: [False] * len(subtasks) for problem_num, subtasks in groups.items()}

    # Track only submissions that improve the final score
    impactful_submissions = []

    # (removed unused random seed and flatten helpers)

    for problem_name, subtasks in groups.items():
        print(f"Processing problem: {problem_name}")

        # Make a copy and reverse the order of subtasks to start with the hardest
        subtasks = subtasks.copy()
        subtasks.sort(reverse=True)

        # Keep track of available subtasks (ones that haven't reached max score)
        available_subtasks = subtasks.copy()

        # Cache for loaded cluster data to avoid reloading
        clusters_cache = {}

        for submission_count in range(50):
            # Filter out completed subtasks
            available_subtasks = [st for i, st in enumerate(subtasks) if not completed_subtasks[problem_name][i]]

            if not available_subtasks:
                print(f"  *** All subtasks in problem {problem_name} have reached maximum score! ***")
                print(f"  *** Stopping early after {submission_count} submissions ***")
                break

            # Pick a subtask to sample from (cycle through available ones)
            current_subtask = available_subtasks[submission_count % len(available_subtasks)]

            # Load cluster data for this subtask if not already cached
            if current_subtask not in clusters_cache:
                try:
                    filepath = f"{data_dir}/{current_subtask}_cluster.jsonl"
                    print(f"    Loading cluster data from {filepath}")
                    clusters_data = load_cluster_data(filepath)
                    clusters = clusters_data
                    # Optional: remove clusters that contain empty-output signature
                    if "REMOVE_EMPTY" in globals() and globals().get("REMOVE_EMPTY"):

                        def _remove_empty_output_clusters(clusters_dict):
                            """Drop clusters that contain the empty-output hash marker."""
                            empty_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
                            filtered = {}
                            for cname, cdata in clusters_dict.items():
                                out = cdata.get("output", [])
                                if isinstance(out, tuple):
                                    out = list(out)
                                has_empty = any(o == empty_hash for o in out)
                                if not has_empty:
                                    filtered[cname] = cdata
                            return filtered

                        clusters = _remove_empty_output_clusters(clusters)
                    # If cluster data is empty or invalid, skip this subtask
                    if not isinstance(clusters, dict) or not clusters:
                        print(f"    Cluster data empty for {filepath}, skipping subtask {current_subtask}")
                        if current_subtask in available_subtasks:
                            available_subtasks.remove(current_subtask)
                        continue

                    # Apply inter-cluster strategy (wins, score, size, random)
                    if strategy == "wins":
                        # Sort by merged tournament_wins descending
                        sorted_clusters = sorted(
                            clusters.items(),
                            key=lambda item: item[1].get("tournament_wins", 0),
                            reverse=True,
                        )
                        print("    Strategy: wins - sorting clusters by merged tournament_wins (higher first)")
                    elif strategy == "score":
                        # Sort by merged tournament_score descending
                        sorted_clusters = sorted(
                            clusters.items(),
                            key=lambda item: item[1].get("tournament_score", 0.0),
                            reverse=True,
                        )
                        print("    Strategy: score - sorting clusters by merged tournament_score (higher first)")
                    elif strategy == "size":
                        # Sort by cluster size (more codes first)
                        def _codes_len_safe(cdict):
                            """Safely return cluster code list length."""
                            codes = cdict.get("codes", [])
                            return len(codes) if isinstance(codes, list) else 0

                        sorted_clusters = sorted(
                            clusters.items(),
                            key=lambda item: _codes_len_safe(item[1]),
                            reverse=True,
                        )
                        print("    Strategy: size - sorting clusters by cluster size (larger first)")
                    elif strategy == "random":
                        # Random order of clusters
                        items = list(clusters.items())
                        random.shuffle(items)
                        sorted_clusters = items
                        print("    Strategy: random - shuffling clusters randomly")
                    else:
                        raise ValueError(
                            f"Unknown inter-cluster strategy: {strategy}. Valid strategies: wins, score, size, random."
                        )

                    # Limit clusters if requested
                    if max_clusters == -1:
                        print(f"    Using all {len(sorted_clusters)} clusters")
                    else:
                        effective_limit = max_clusters
                        sorted_clusters = sorted_clusters[:effective_limit]
                        print(
                            f"    Using top {min(effective_limit, len(sorted_clusters))} clusters (out of {len(clusters)} total)"
                        )

                    # Sort solutions within each cluster by chosen intra-cluster strategy.
                    # Supported (aligned with ICPC script): longest, score, wins,
                    # shortest, first, random.
                    def _safe_int(v, default=0):
                        """Convert value to int with fallback default."""
                        try:
                            return int(v)
                        except Exception:
                            return default

                    def _safe_float(v, default=0.0):
                        """Convert value to float with fallback default."""
                        try:
                            return float(v)
                        except Exception:
                            return default

                    for cluster_name, cluster_data in sorted_clusters:
                        codes = cluster_data.get("codes", [])
                        filtered_codes = [c for c in codes if isinstance(c, dict)]
                        if not filtered_codes:
                            cluster_data["codes"] = []
                            continue

                        intra = in_cluster_strategy
                        if intra == "longest":
                            sorted_codes = sorted(
                                filtered_codes,
                                key=lambda c: _safe_int(c.get("tokens", 0), 0),
                                reverse=True,
                            )
                        elif intra == "shortest":
                            sorted_codes = sorted(
                                filtered_codes,
                                key=lambda c: _safe_int(c.get("tokens", 0), 0),
                            )
                        elif intra == "score":
                            # Sort by per-solution tournament_score desc; tie-break by tokens desc
                            sorted_codes = sorted(
                                filtered_codes,
                                key=lambda c: (
                                    _safe_float(c.get("tournament_score", 0.0), 0.0),
                                    _safe_int(c.get("tokens", 0), 0),
                                ),
                                reverse=True,
                            )
                        elif intra == "wins":
                            # Sort by per-solution tournament_wins desc; tie-break by tokens desc
                            sorted_codes = sorted(
                                filtered_codes,
                                key=lambda c: (
                                    _safe_int(c.get("tournament_wins", 0), 0),
                                    _safe_int(c.get("tokens", 0), 0),
                                ),
                                reverse=True,
                            )
                        elif intra == "random":
                            sorted_codes = list(filtered_codes)
                            random.shuffle(sorted_codes)
                        else:  # "first" or unknown → keep input order
                            sorted_codes = list(filtered_codes)

                        cluster_data["codes"] = sorted_codes

                    # Create a dictionary from the sorted list of clusters
                    sorted_clusters_dict = {k: v for k, v in sorted_clusters}

                    # If no clusters remain after sorting/limiting, skip subtask
                    if not sorted_clusters_dict:
                        print(f"    No clusters available after filtering for subtask {current_subtask}")
                        if current_subtask in available_subtasks:
                            available_subtasks.remove(current_subtask)
                        continue

                    # Create an iterator to go through solutions in a round-robin fashion
                    solution_iterator = get_solution_iterator(sorted_clusters_dict)

                    clusters_cache[current_subtask] = solution_iterator

                except FileNotFoundError:
                    print(f"Could not find cluster file for subtask {current_subtask}")
                    # Remove this subtask from available list
                    if current_subtask in available_subtasks:
                        available_subtasks.remove(current_subtask)
                    continue

            # Get the next solution from the current subtask
            solution_iterator = clusters_cache[current_subtask]
            cluster_name, solution_index, solution = next(solution_iterator)

            # Record the grade for this solution
            grade = solution["grade"]
            tokens = solution.get("tokens", "N/A")

            # Debug: Print detailed info about what we're getting
            print(
                f"  Submission {submission_count + 1}: Problem: {problem_name}, Current Subtask: {current_subtask}, Available Subtasks: {available_subtasks}, Cluster: {cluster_name}, Solution Index: {solution_index}, Tokens Generated: {tokens}, Grade: {grade}"
            )

            # Update max scores for each subtask
            # For IOI 25 problems 7-12 and 13-18, only use relevant part of grade array
            grade_slice = get_grade_slice_for_problem(problem_name, dataset)
            relevant_grade = grade[grade_slice]

            # Track improvements caused by this specific submission
            impacted_details = []

            for i, score in enumerate(relevant_grade):
                if score > max_scores[problem_name][i]:
                    max_scores[problem_name][i] = score
                    # Find the original subtask number
                    original_subtask_index = len(subtasks) - i - 1
                    subtask_number = subtasks[original_subtask_index]
                    print(f"      New max score for subtask {subtask_number}: {score}")
                    impacted_details.append({"subtask_number": subtask_number, "new_score": score})

                    # Check if this subtask has reached its maximum possible score
                    max_possible = get_max_score_for_subtask(subtask_number, dataset)
                    if max_possible > 0 and score >= max_possible:
                        # Use the original subtask index for consistency with filtering
                        completed_subtasks[problem_name][original_subtask_index] = True
                        print(f"      *** Subtask {subtask_number} has reached maximum score! ***")

            # If this submission improved any subtask, record it
            if impacted_details:
                impactful_submissions.append(
                    {
                        "dataset": dataset,
                        "data_dir": data_dir,
                        "problem_group": problem_name,
                        "selected_subtask": current_subtask,
                        "code": solution["code"],
                        "cluster_name": cluster_name,
                        "solution_index": solution_index,
                        "tokens": tokens,
                        "impacted_subtasks": impacted_details,
                        "grade": relevant_grade,
                        "subtask_file": f"{data_dir}/{current_subtask}_cluster.jsonl",
                    }
                )

    # Write impactful submissions to output file
    try:
        output_path = "submission.jsonl"
        with open(output_path, "w") as f:
            for record in impactful_submissions:
                f.write(json.dumps(record) + "\n")
        print(f"Wrote {len(impactful_submissions)} impactful submissions to {output_path}")
    except Exception as e:
        print(f"Failed to write impactful submissions: {e}")

    # Print the final max scores
    for problem_name, scores in max_scores.items():
        print(f"Max scores for problem {problem_name}: {scores}")

    # Calculate and print the total score
    total_score = sum(score for scores in max_scores.values() for score in scores)
    print(f"Total score: {total_score}")

    return max_scores


def calculate_theoretical_max_score(submission_scores=None, dataset="ioi24", data_subdir=None):
    """Calculates the theoretical maximum score without the 50-submission limit."""
    groups_24 = {
        "1-7": list(range(1, 8)),
        "9-10": [9, 10],
        "12-18": list(range(12, 19)),
        "20-25": list(range(20, 26)),
        "27-34": list(range(27, 35)),
        "36-40": list(range(36, 41)),
    }
    groups_25 = {
        "1-6": list(range(1, 7)),
        "7-12": list(range(7, 13)),  # First part of problem 2 (look at first 6 scores)
        "13-18": list(range(13, 19)),  # Second part of problem 2 (look at second 6 scores)
        "19-24": list(range(19, 25)),
        "25-31": list(range(25, 32)),
        "32-33": list(range(32, 34)),
        "34-39": list(range(34, 40)),
    }

    # Select the appropriate dataset configuration
    if dataset == "ioi24":
        groups = groups_24
        default_dir = "ioi_24/0"
    elif dataset == "ioi25":
        groups = groups_25
        default_dir = "ioi_25/1"
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'ioi24' or 'ioi25'")

    data_dir = data_subdir if data_subdir else default_dir

    # Store original subtasks order for each problem group (before any sorting)
    original_subtasks_order = {problem_num: subtasks.copy() for problem_num, subtasks in groups.items()}

    # Tracking for cluster files
    cluster_max_scores = {problem_num: [0.0] * len(subtasks) for problem_num, subtasks in groups.items()}

    for problem_name, subtasks in groups.items():
        # Check each subtask in the group
        for subtask in subtasks:
            # Try to load cluster file for this subtask
            cluster_filepath = f"{data_dir}/{subtask}_cluster.jsonl"
            try:
                clusters = load_cluster_data(cluster_filepath)

                for cluster_data in clusters.values():
                    for solution in cluster_data["codes"]:
                        grade = solution["grade"]
                        # For IOI 25 problems 7-12 and 13-18, only use relevant part of grade array
                        grade_slice = get_grade_slice_for_problem(problem_name, dataset)
                        relevant_grade = grade[grade_slice]

                        for i, score in enumerate(relevant_grade):
                            if score > cluster_max_scores[problem_name][i]:
                                cluster_max_scores[problem_name][i] = score
            except FileNotFoundError:
                pass

            # Skip loading individual jsonl files (removed)

    # Calculate totals for cluster source
    total_cluster_max = sum(score for scores in cluster_max_scores.values() for score in scores)

    # Comparison table
    print("\n" + "=" * 100)
    print("COMPARISON: CLUSTER vs SUBMISSION vs KNOWN MAX")
    print("=" * 100)
    for problem_name, subtasks in groups.items():
        print(f"Problem {problem_name}:")
        print(f"  {'Subtask':<8} {'Cluster':<8} {'Submission':<10} {'Known Max':<10} {'Status'}")
        print(f"  {'-' * 8} {'-' * 8} {'-' * 10} {'-' * 10} {'-' * 6}")
        for i, subtask in enumerate(subtasks):
            # For problem groups like '1-7', the grade array is [score_1, score_2, ..., score_7]
            # For problem groups like '9-10', the grade array is [score_9, score_10]
            # So we need to find the position of this subtask within the original subtasks list
            grade_index = original_subtasks_order[problem_name].index(subtask)

            cluster_val = cluster_max_scores[problem_name][grade_index]
            known_max = get_max_score_for_subtask(subtask, dataset)

            # Get submission score if available
            if submission_scores and problem_name in submission_scores:
                submission_val = submission_scores[problem_name][grade_index]
            else:
                submission_val = 0.0

            # Status: check if each source reaches known max
            cluster_ok = "✓" if cluster_val >= known_max else "✗"
            submission_ok = "✓" if submission_val >= known_max else "✗"
            status = f"C:{cluster_ok} S:{submission_ok}"

            print(f"  {subtask:<8} {cluster_val:<8} {submission_val:<10} {known_max:<10} {status}")
        print()

    print("\n" + "=" * 60)
    print(f"THEORETICAL MAXIMUM FROM CLUSTER FILES: {total_cluster_max}")
    print("=" * 60)
    for problem_name, scores in cluster_max_scores.items():
        print(f"  Problem {problem_name}: {scores}")

    # Removed individual-file theoretical maximum reporting

    # Calculate submission total if available
    if submission_scores:
        total_submission_score = sum(score for scores in submission_scores.values() for score in scores)
        print("\n" + "=" * 60)
        print(f"ACTUAL SUBMISSION SCORE (50 submissions limit): {total_submission_score}")
        print("=" * 60)
        for problem_name, scores in submission_scores.items():
            print(f"  Problem {problem_name}: {scores}")


if __name__ == "__main__":
    # Parse command line arguments
    STRATEGY = "wins"  # Default inter-cluster strategy
    MAX_CLUSTERS = -1  # Default to top 5 clusters
    DATASET = "ioi24"  # Default to IOI 2024

    # Simple argument parsing
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        # Explicit inter-cluster strategy (backwards-compatible alias: --strategy)
        if arg.startswith("--inter-strategy=") or arg.startswith("--strategy="):
            try:
                strategy = arg.split("=", 1)[1].lower()
                allowed_inter = ["wins", "score", "size", "random"]
                if strategy not in allowed_inter:
                    print(f"Invalid inter-cluster strategy: {strategy}")
                    print("Valid inter strategies: wins, score, size, random")
                    sys.exit(1)
                STRATEGY = strategy
            except IndexError:
                print("Invalid --inter-strategy/--strategy format. Use --inter-strategy=STRATEGY_NAME")
                sys.exit(1)
        # Cluster selection options
        elif arg.startswith("--max-clusters="):
            try:
                MAX_CLUSTERS = int(arg.split("=")[1])
            except (IndexError, ValueError):
                print("Invalid --max-clusters value. Use --max-clusters=N (or -1 for all)")
                sys.exit(1)
        elif arg in ["--all-clusters", "--all"]:
            MAX_CLUSTERS = -1
        # Dataset options
        elif arg.startswith("--dataset="):
            try:
                dataset = arg.split("=")[1].lower()
                if dataset in ["ioi24", "ioi25"]:
                    DATASET = dataset
                else:
                    print(f"Invalid dataset: {dataset}")
                    print("Valid datasets: ioi24, ioi25")
                    sys.exit(1)
            except IndexError:
                print("Invalid --dataset format. Use --dataset=DATASET_NAME")
                sys.exit(1)
        elif arg in ["--ioi24"]:
            DATASET = "ioi24"
        elif arg in ["--ioi25"]:
            DATASET = "ioi25"

        elif arg.startswith("--data-subdir=") or arg.startswith("--folder="):
            try:
                DATA_SUBDIR = arg.split("=")[1]
            except IndexError:
                print("Invalid --data-subdir/--folder format. Use --data-subdir=ioi_25/0")
                sys.exit(1)
        # Intra-cluster ordering (backwards-compatible alias: --in-cluster-strategy)
        elif arg.startswith("--intra-strategy=") or arg.startswith("--in-cluster-strategy="):
            try:
                IN_CLUSTER_STRATEGY = arg.split("=", 1)[1].lower()
                allowed_intra = ["longest", "score", "wins", "shortest", "first", "random"]
                if IN_CLUSTER_STRATEGY not in allowed_intra:
                    print("Invalid --intra-strategy/--in-cluster-strategy.")
                    print("Valid intra strategies: longest, score, wins, shortest, first, random")
                    sys.exit(1)
            except IndexError:
                print(
                    "Invalid --intra-strategy/--in-cluster-strategy format. "
                    "Use --intra-strategy=longest|score|wins|shortest|first|random"
                )
                sys.exit(1)
        elif arg.startswith("--remove-empty"):
            REMOVE_EMPTY = True
        else:
            print("Usage: python submission_IOI.py [OPTIONS]")
            print("Dataset selection:")
            print("  --ioi24:               Use IOI 2024 dataset (default)")
            print("  --ioi25:               Use IOI 2025 dataset")
            print("  --dataset=DATASET:     Specify dataset (ioi24 or ioi25)")
            print("  --data-subdir=PATH     Override data folder, e.g., ioi_25/0")
            print("  --clusters-per-subtask=N  Scale cluster limit by #subtasks per problem")
            print("Inter-cluster strategy selection (between clusters):")
            print("  wins:    Sort clusters by merged tournament_wins (default)")
            print("  score:   Sort clusters by merged tournament_score")
            print("  size:    Sort clusters by cluster size (#codes)")
            print("  random:  Shuffle clusters randomly")
            print("  --inter-strategy=STRATEGY: Explicitly specify inter-cluster strategy (alias: --strategy)")
            print("Intra-cluster strategy selection (within a cluster):")
            print("  --intra-strategy=longest   Pick longest solutions first (by tokens)")
            print("  --intra-strategy=shortest  Pick shortest solutions first (by tokens)")
            print("  --intra-strategy=score     Pick by per-solution tournament_score")
            print("  --intra-strategy=wins      Pick by per-solution tournament_wins")
            print("  --intra-strategy=first     Keep original order")
            print("  --intra-strategy=random    Random order within cluster")
            print("Cluster selection options:")
            print("  --max-clusters=N:      Use top N clusters (default: 5)")
            print("  --all:                 Use all available clusters")
            print("Examples:")
            print("  python submission_IOI.py --ioi25 --inter-strategy=wins --intra-strategy=longest")
            print("  python submission_IOI.py --dataset=ioi24 --inter-strategy=score --max-clusters=10")
            print("  python submission_IOI.py --ioi25 --inter-strategy=size --all --intra-strategy=wins")
            sys.exit(1)
        i += 1

    print("Configuration:")
    print(f"  Dataset: {DATASET.upper()}")
    print(f"  Inter-cluster strategy: {STRATEGY}")
    print(f"  Max clusters per subtask: {'All' if MAX_CLUSTERS == -1 else MAX_CLUSTERS}")
    if "DATA_SUBDIR" in globals():
        print(f"  Data subdir override: {DATA_SUBDIR}")
    if "IN_CLUSTER_STRATEGY" in globals():
        print(f"  Intra-cluster (within-cluster) strategy: {IN_CLUSTER_STRATEGY}")
    if len(sys.argv) > 1:
        print(f"  (command line args: {' '.join(sys.argv[1:])})")
    print()

    submission_max_scores = run_submission(
        strategy=STRATEGY,
        max_clusters=MAX_CLUSTERS,
        dataset=DATASET,
        data_subdir=globals().get("DATA_SUBDIR"),
        in_cluster_strategy=globals().get("IN_CLUSTER_STRATEGY", "longest"),
    )
    calculate_theoretical_max_score(submission_max_scores, dataset=DATASET, data_subdir=globals().get("DATA_SUBDIR"))
