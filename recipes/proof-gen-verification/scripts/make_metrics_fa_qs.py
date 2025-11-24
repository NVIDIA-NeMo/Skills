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

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from nemo_skills.evaluation.math_grader import extract_answer, math_equal


# For proof prompts, it keeps adding stuff that is not digits to the answer.
# and our parsing fails.
def keep_only_digits(answer):
    if answer is None:
        return None
    answer = answer.split("=")[-1]
    return "".join(filter(str.isdigit, answer))


def compute_majority_k(original_proofs_list, expected_answer):
    """Compute majority@k answer and correctness from original proofs list."""
    original_answers = []
    for proof_data in original_proofs_list:
        proof_text = proof_data["proof"]
        answer = extract_answer(proof_text)
        # Keep only digits of answer
        if answer is not None:  # Discard None answers
            answer = keep_only_digits(answer)
            original_answers.append(answer)

    k = len(original_proofs_list)
    answer_counts = Counter(original_answers)
    try:
        maj_k_answer = answer_counts.most_common(1)[0][0]
        maj_k_correct = math_equal(maj_k_answer, expected_answer) if maj_k_answer and expected_answer else False
        if maj_k_answer != expected_answer:
            print(f"Majority answer {maj_k_answer} is not equal to expected answer {expected_answer}")
    except IndexError:
        print(f"No majority answer found for {original_proofs_list}")
        maj_k_answer = None
        maj_k_correct = False

    return maj_k_answer, maj_k_correct, k


def compute_pass_at_1(original_proofs_list, expected_answer):
    """Compute pass@1 - average correctness across all individual attempts."""
    k = len(original_proofs_list)

    if k == 0:
        return 0.0, k

    correct_count = 0
    for proof_data in original_proofs_list:
        proof_text = proof_data["proof"]
        answer = extract_answer(proof_text)

        # Keep only digits of answer
        if answer is not None:
            answer = keep_only_digits(answer)
            if math_equal(answer, expected_answer):
                correct_count += 1

    # Return average correctness (fraction of correct attempts)
    pass_at_1_score = correct_count / k
    return pass_at_1_score, k


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="Input directory containing output-rs*.jsonl files")
    parser.add_argument("--output_file", help="Output path for metrics.json")
    parser.add_argument("--num_seeds", type=int, help="Number of seed files to process")

    args = parser.parse_args()
    all_samples = []

    # Process files for seeds 0 to num_seeds-1
    for i in range(args.num_seeds):
        input_file = Path(args.input_dir) / f"output-rs{i}.jsonl"

        with open(input_file, "r") as f:
            data = [json.loads(line) for line in f]

        for sample in data:
            proof_judgements_results = sample["proof_judgements_results"]

            # genselect_best_proof = max(proof_judgements_results, key=lambda x: x["genselected_judgements_score"])[
            #     "proof"
            # ]
            # genselect_best_answer = extract_answer(genselect_best_proof)
            # genselect_best_answer = keep_only_digits(genselect_best_answer)
            llm_as_judge_best_proof = max(proof_judgements_results, key=lambda x: x["judgements_score"])["proof"]
            llm_as_judge_best_answer = extract_answer(llm_as_judge_best_proof)
            llm_as_judge_best_answer = keep_only_digits(llm_as_judge_best_answer)
            expected_answer = sample["expected_answer"]

            llm_judge_correct = (
                math_equal(llm_as_judge_best_answer, expected_answer) if llm_as_judge_best_answer else False
            )
            # genselect_correct = math_equal(genselect_best_answer, expected_answer) if genselect_best_answer else False
            if not llm_judge_correct:
                print(f"LLM judge answer {llm_as_judge_best_answer} is not equal to expected answer {expected_answer}")

            maj_k_answer, maj_k_correct, k = compute_majority_k(sample["original_proofs_list"], expected_answer)
            pass_at_1_score, k_pass = compute_pass_at_1(sample["original_proofs_list"], expected_answer)

            all_samples.append(
                {
                    "llm_judge_answer": llm_as_judge_best_answer,
                    # "genselect_answer": genselect_best_answer,
                    "expected_answer": expected_answer,
                    "llm_judge_correct": llm_judge_correct,
                    # "genselect_correct": genselect_correct,
                    "maj_k_answer": maj_k_answer,
                    "maj_k_correct": maj_k_correct,
                    "pass_at_1_score": pass_at_1_score,
                    "k": k,
                }
            )

    # Compute metrics
    total = len(all_samples)
    llm_judge_correct = sum(1 for s in all_samples if s["llm_judge_correct"])
    # genselect_correct = sum(1 for s in all_samples if s["genselect_correct"])
    maj_k_correct = sum(1 for s in all_samples if s["maj_k_correct"])
    pass_at_1_total_score = sum(s["pass_at_1_score"] for s in all_samples)

    llm_judge_accuracy = llm_judge_correct / total if total > 0 else 0
    # genselect_accuracy = genselect_correct / total if total > 0 else 0
    maj_k_accuracy = maj_k_correct / total if total > 0 else 0
    pass_at_1_accuracy = pass_at_1_total_score / total if total > 0 else 0

    # Assert all k values are equal
    k_values = [s["k"] for s in all_samples]
    k = list(set(k_values))

    metrics = {
        "llm_judge": {"accuracy": llm_judge_accuracy * 100, "correct": llm_judge_correct, "total": total},
        # "genselect": {"accuracy": genselect_accuracy * 100, "correct": genselect_correct, "total": total},
        "maj_k": {"accuracy": maj_k_accuracy * 100, "correct": maj_k_correct, "total": total, "k_values": k},
        "pass_at_1": {
            "accuracy": pass_at_1_accuracy * 100,
            "total": total,
            "k_values": k,
        },
    }

    # Save metrics
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"LLM Judge: {llm_judge_correct}/{total} = {llm_judge_accuracy:.3f}")
    # print(f"GenSelect: {genselect_correct}/{total} = {genselect_accuracy:.3f}")
    print(f"Maj@{k}: {maj_k_correct}/{total} = {maj_k_accuracy:.3f}")
    print(f"Pass@1: {pass_at_1_total_score:.1f}/{total} = {pass_at_1_accuracy:.3f}")
    print(f"Saved to: {args.output_file}")


if __name__ == "__main__":
    main()
