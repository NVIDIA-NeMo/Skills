#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Generic BON metrics computation script.
Computes best-of-n metrics from evaluation results for llm-as-a-judge (binary and scoring) and genselect methods.
"""

import argparse
import hashlib
import json
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np


def compute_metrics_for_seed(seed_idx: int, data_list: list[dict], eval_type: str, num_shuffles: int) -> dict:
    print(f"Processing seed {seed_idx + 1}...")
    if eval_type.startswith("llm_as_judge"):
        metrics = compute_llm_as_judge_metrics(data_list, num_shuffles)
    elif eval_type == "genselect":
        metrics = compute_genselect_metrics(data_list, num_shuffles)
    else:
        raise ValueError(f"Unknown evaluation type: {eval_type}")

    print(f"Completed seed {seed_idx + 1}")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Generic BON metrics computation")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing evaluation results")
    parser.add_argument("--output_file", type=str, required=True, help="Output file for metrics")
    parser.add_argument(
        "--num_shuffles", type=int, default=1000, help="Number of random shuffles for computing metrics"
    )
    parser.add_argument(
        "--num_processes", type=int, default=None, help="Number of processes to use (default: number of CPU cores)"
    )

    args = parser.parse_args()

    # Note: We use deterministic shuffles based on (problem_id, n, shuffle_idx) hashing
    # This ensures that llm_as_judge and genselect use the exact same shuffles for fair comparison

    # Load evaluation results by seed (each jsonl file is one seed)
    seeds_data, eval_type = load_evaluation_results_by_seed(args.input_dir)

    # Determine number of processes to use
    num_processes = args.num_processes if args.num_processes is not None else cpu_count()
    num_processes = min(num_processes, len(seeds_data))

    print(f"Using {num_processes} processes to compute metrics for {len(seeds_data)} seeds")

    # Compute metrics for each seed in parallel
    worker_fn = partial(compute_metrics_for_seed, eval_type=eval_type, num_shuffles=args.num_shuffles)

    with Pool(processes=num_processes) as pool:
        seed_metrics = pool.starmap(
            worker_fn, [(seed_idx, data_list) for seed_idx, data_list in enumerate(seeds_data)]
        )

    # Aggregate metrics across seeds (mean and std)
    aggregated_metrics = aggregate_metrics_across_seeds(seed_metrics)

    # Save metrics
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(aggregated_metrics, f, indent=2)

    print(f"Metrics saved to {output_path}")


def load_evaluation_results_by_seed(input_dir: str) -> tuple[list[list[dict]], str]:
    input_path = Path(input_dir)
    seeds_data = []
    eval_type = None

    # Find all output files
    output_files = sorted(input_path.rglob("output-*.jsonl"))
    assert len(output_files) > 0, f"No output files found in {input_dir}"

    for file_path in output_files:
        assert "chunk" not in file_path.name, "File is chunked, not completed yet."

        seed_data = []
        with open(file_path, "r") as f:
            for line in f:
                datapoint = json.loads(line)

                # Check if this datapoint has evaluation results
                current_eval_type = datapoint["evaluation_type"]

                if eval_type is None:
                    eval_type = current_eval_type
                else:
                    assert eval_type == current_eval_type, f"Mixed eval types: {eval_type} and {current_eval_type}"

                seed_data.append(datapoint)

        seeds_data.append(seed_data)
        print(f"Loaded seed {len(seeds_data)} from {file_path.name}: {len(seed_data)} datapoints")

    assert eval_type is not None, "No evaluation data found"
    assert len(seeds_data) > 0, "No seeds with evaluation data found"

    print(f"Total seeds: {len(seeds_data)}, eval_type: {eval_type}")
    return seeds_data, eval_type


def expert_score_to_correctness(score: float) -> int:
    return 1 if score >= 6 else 0


def aggregate_dict_across_seeds(values_across_seeds: list) -> dict:
    if len(values_across_seeds) == 0:
        return {}

    first_value = values_across_seeds[0]

    if isinstance(first_value, dict):
        result = {}
        for key in first_value.keys():
            nested_values = [v[key] for v in values_across_seeds if key in v]
            result[key] = aggregate_dict_across_seeds(nested_values)
        return result
    elif isinstance(first_value, (int, float)):
        return {
            "mean": float(np.mean(values_across_seeds)),
            "std": float(np.std(values_across_seeds)) if len(values_across_seeds) > 1 else 0.0,
        }
    else:
        return first_value


def aggregate_metrics_across_seeds(seed_metrics: list[dict]) -> dict:
    if len(seed_metrics) == 0:
        return {}

    result = aggregate_dict_across_seeds(seed_metrics)
    result["num_seeds"] = len(seed_metrics)
    return result


def compute_llm_as_judge_metrics(data_list: list[dict], num_shuffles: int) -> dict:
    metrics = {}
    max_n_proofs = max(len(dp["proofs"]) for dp in data_list)
    max_judgements = max(
        len(proof_data["scores"]) for dp in data_list for proof_data in dp["evaluation_results"]["proofs_judgements"]
    )

    # Initialize storage for ablation metrics
    judgement_ablation = {f"num_judgements={num_j}": {} for num_j in range(1, max_judgements + 1)}

    # Metrics as a function of number of proofs considered
    for n in range(1, max_n_proofs + 1):
        avg_best_scores = []
        avg_best_correctness = []

        # Initialize storage for ablation at this n
        ablation_scores = {num_j: [] for num_j in range(1, max_judgements + 1)}
        ablation_correctness = {num_j: [] for num_j in range(1, max_judgements + 1)}

        # For each problem
        for datapoint in data_list:
            proofs = datapoint["proofs"]
            expert_ratings = datapoint["expert_ratings"]
            eval_results = datapoint["evaluation_results"]["proofs_judgements"]
            problem_id = datapoint["problem_id"]

            # Run multiple shuffles with deterministic seed based on problem_id and n
            # This ensures same shuffles are used for this problem across different evaluation methods
            for shuffle_idx in range(num_shuffles):
                # Create deterministic seed for this shuffle
                hash_input = f"{problem_id}_{n}_{shuffle_idx}".encode("utf-8")
                shuffle_seed = int(hashlib.md5(hash_input).hexdigest(), 16) % (2**32)
                shuffle_rng = np.random.RandomState(shuffle_seed)

                # Randomly select n proofs
                indices = shuffle_rng.choice(len(proofs), size=min(n, len(proofs)), replace=False)

                # Generate a single permutation of judgement indices for all proofs
                judgement_permutation = shuffle_rng.permutation(max_judgements)

                # Main computation: Find the proof with highest average score among selected
                best_idx_in_subset = max(range(len(indices)), key=lambda i: eval_results[indices[i]]["average_score"])
                best_original_idx = indices[best_idx_in_subset]

                # Get the expert rating for this proof
                best_score = expert_ratings[best_original_idx]
                best_correctness = expert_score_to_correctness(best_score)

                avg_best_scores.append(best_score)
                avg_best_correctness.append(best_correctness)

                # Ablation: compute metrics for different numbers of judgements
                for num_judgements in range(1, max_judgements + 1):
                    # For each selected proof, compute average score using only num_judgements random judgements
                    proof_avg_scores = []
                    for idx in indices:
                        proof_data = eval_results[idx]
                        all_scores = proof_data["scores"]

                        if len(all_scores) < num_judgements:
                            continue

                        # Use the same permutation to select judgements
                        selected_indices = judgement_permutation[:num_judgements]
                        selected_scores = [all_scores[i] for i in selected_indices if i < len(all_scores)]

                        # Filter out None values for averaging
                        valid_scores = [s for s in selected_scores if s is not None]
                        avg_score = float(np.mean(valid_scores)) if len(valid_scores) > 0 else 0.0
                        proof_avg_scores.append((idx, avg_score))

                    # Find the proof with highest average score among selected
                    best_idx_in_subset_ablation = max(
                        range(len(proof_avg_scores)), key=lambda i: proof_avg_scores[i][1]
                    )
                    best_original_idx_ablation = proof_avg_scores[best_idx_in_subset_ablation][0]

                    # Get the expert rating for this proof
                    ablation_best_score = expert_ratings[best_original_idx_ablation]
                    ablation_best_correctness = expert_score_to_correctness(ablation_best_score)

                    ablation_scores[num_judgements].append(ablation_best_score)
                    ablation_correctness[num_judgements].append(ablation_best_correctness)

        metrics[f"n={n}"] = {
            "avg_best_score": float(np.mean(avg_best_scores)) if avg_best_scores else 0.0,
            "avg_best_correctness": float(np.mean(avg_best_correctness)) if avg_best_correctness else 0.0,
            "num_samples": len(avg_best_scores),
        }

        # Store ablation metrics for this n
        for num_judgements in range(1, max_judgements + 1):
            judgement_ablation[f"num_judgements={num_judgements}"][f"n={n}"] = {
                "avg_best_score": float(np.mean(ablation_scores[num_judgements]))
                if ablation_scores[num_judgements]
                else 0.0,
                "avg_best_correctness": float(np.mean(ablation_correctness[num_judgements]))
                if ablation_correctness[num_judgements]
                else 0.0,
                "num_samples": len(ablation_scores[num_judgements]),
            }

    metrics["judgement_ablation"] = judgement_ablation

    return metrics


def compute_genselect_metrics(data_list: list[dict], num_shuffles: int) -> dict:
    metrics = {}
    max_n_proofs = max(len(dp["proofs"]) for dp in data_list)

    # Metrics as a function of number of proofs considered
    for n in range(1, max_n_proofs + 1):
        avg_best_scores = []
        avg_best_correctness = []

        # For each problem
        for datapoint in data_list:
            proofs = datapoint["proofs"]
            expert_ratings = datapoint["expert_ratings"]
            pairwise_comparisons = datapoint["evaluation_results"]["pairwise_comparisons"]
            problem_id = datapoint["problem_id"]

            # Run multiple shuffles with deterministic seed based on problem_id and n
            # This ensures same shuffles are used for this problem across different evaluation methods
            for shuffle_idx in range(num_shuffles):
                # Create deterministic seed for this shuffle (same as llm_as_judge)
                hash_input = f"{problem_id}_{n}_{shuffle_idx}".encode("utf-8")
                shuffle_seed = int(hashlib.md5(hash_input).hexdigest(), 16) % (2**32)
                shuffle_rng = np.random.RandomState(shuffle_seed)

                # Randomly select n proofs
                selected_indices = shuffle_rng.choice(len(proofs), size=min(n, len(proofs)), replace=False)
                selected_set = set(selected_indices)

                # Count wins for each proof in this subset
                wins = {idx: 0 for idx in selected_indices}

                for comparison in pairwise_comparisons:
                    idx1 = comparison["proof1_idx"]
                    idx2 = comparison["proof2_idx"]
                    winner_idx = comparison["winner_idx"]

                    if idx1 in selected_set and idx2 in selected_set:
                        if winner_idx is None:
                            # Parsing issue, or no winner found
                            wins[idx1] += 0.5
                            wins[idx2] += 0.5
                        else:
                            wins[winner_idx] += 1

                # Find the proof with most wins
                best_idx_in_subset = max(selected_indices, key=lambda idx: wins[idx])

                # Get the expert rating for this proof
                best_score = expert_ratings[best_idx_in_subset]
                best_correctness = expert_score_to_correctness(best_score)

                avg_best_scores.append(best_score)
                avg_best_correctness.append(best_correctness)

        metrics[f"n={n}"] = {
            "avg_best_score": float(np.mean(avg_best_scores)) if avg_best_scores else 0.0,
            "avg_best_correctness": float(np.mean(avg_best_correctness)) if avg_best_correctness else 0.0,
            "num_samples": len(avg_best_scores),
        }

    return metrics


if __name__ == "__main__":
    main()
