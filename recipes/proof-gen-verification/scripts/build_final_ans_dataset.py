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
Best of N evaluation script for OPC dataset.
This script evaluates Best of N performance on generated solutions.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def gather_problem_solutions(input_dir):
    input_path = Path(input_dir)
    results = []
    for model_dir in input_path.iterdir():
        if not model_dir.is_dir():
            continue
        model_id = model_dir.name
        # Look for any output_*.jsonl files
        for output_file in model_dir.glob("output*.jsonl"):
            with open(output_file, "r") as f:
                for line in f:
                    data = json.loads(line)
                    results.append({**data, "model_id": model_id})
    if not results:
        print(f"Warning: No output files found in {input_dir}")
    return results


def filter_problem_solutions(all_data, n_pos_neg, reference_model=None, reference_model_threshold=None):
    filtered_data = []
    model_problem_positives = defaultdict(list)
    model_problem_negatives = defaultdict(list)
    problem_to_source = {}
    problem_to_expected_answer = {}
    # Track per-problem success stats for the reference model, if provided
    reference_success_counts = defaultdict(int)
    reference_total_counts = defaultdict(int)
    print(f"Found {len(all_data)} problems, Filtering...")
    for data in all_data:
        problem_to_source[data["problem"]] = data["source"]
        problem_to_expected_answer[data["problem"]] = data["expected_answer"]
        model_problem_positives[(data["model_id"], data["problem"])].extend(data["positives"])
        model_problem_negatives[(data["model_id"], data["problem"])].extend(data["negatives"])
        if reference_model is not None and data["model_id"] == reference_model:
            # Count successes and totals for the reference model per problem
            reference_success_counts[data["problem"]] += len(data["positives"])
            reference_total_counts[data["problem"]] += len(data["positives"]) + len(data["negatives"])
    for model_id, problem in model_problem_positives.keys():
        # If reference filtering is enabled, skip problems where the reference model solved too often
        if reference_model is not None and reference_model_threshold is not None:
            total = reference_total_counts[problem]
            success = reference_success_counts[problem]
            keep_problem = True
            success_ratio = success / total
            keep_problem = success_ratio < reference_model_threshold
            # If there were no samples for the reference model for this problem, we keep it by default
            if not keep_problem:
                continue
        # Keep min(positives, negatives, n_pos_neg) for both positives and negatives
        positives = model_problem_positives[(model_id, problem)]
        negatives = model_problem_negatives[(model_id, problem)]
        keep_cnt = min(n_pos_neg, len(positives), len(negatives))
        if keep_cnt > 0:
            positives = random.sample(positives, keep_cnt)
            negatives = random.sample(negatives, keep_cnt)
            for lst, judgement in zip([positives, negatives], ["Judgement: Yes", "Judgement: No"]):
                for dp in lst:
                    filtered_data.append(
                        {
                            "problem": problem,
                            "proof": dp,
                            "expected_judgement": judgement,
                            "subset_for_metrics": "final_answer",
                            "source": problem_to_source[problem],
                            "model_id": model_id,
                            "metadata": {
                                "expected_answer": problem_to_expected_answer[problem],
                            },
                        }
                    )

    print(f"Keeping {len(filtered_data)} problems.")
    return filtered_data


def main():
    """Main function for Best of N evaluation."""
    parser = argparse.ArgumentParser(description="Best of N evaluation for OPC dataset")
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Input directory containing model inference results"
    )
    parser.add_argument("--output_file", type=str, required=True, help="Output file to save the final answer dataset")
    parser.add_argument(
        "--n_pos_neg", type=int, required=True, help="Number of positive and negative examples to sample"
    )
    parser.add_argument(
        "--reference_model",
        type=str,
        default=None,
        help="Model id to use as reference for per-problem success ratio filtering.",
    )
    parser.add_argument(
        "--reference_model_threshold",
        type=float,
        default=None,
        help="Keep only problems where reference model success ratio (pos/(pos+neg)) is below this threshold.",
    )
    args = parser.parse_args()

    all_data = gather_problem_solutions(args.input_dir)
    filtered_data = filter_problem_solutions(
        all_data,
        args.n_pos_neg,
        reference_model=args.reference_model,
        reference_model_threshold=args.reference_model_threshold,
    )
    with open(args.output_file, "w") as f:
        for data in filtered_data:
            f.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    main()
