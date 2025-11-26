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
import random
from collections import defaultdict
from pathlib import Path

import datasets
from transformers import AutoTokenizer

JUDGEMENT_YES = "Judgement: Yes"
JUDGEMENT_NO = "Judgement: No"

MAX_QWEN_TOKENS = 10000
qwen3_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


def prepare_verification_data(output_path):
    print("Preparing Verification data.")
    processed_data = load_hf_data(split="train")

    for item in processed_data:
        item["problem"] = item["problem"].strip()
        item["proof"] = item["proof"].strip()
    random.seed(42)
    random.shuffle(processed_data)

    # Filter out too long proofs
    filtered_processed_data = [
        item for item in processed_data if len(qwen3_tokenizer.encode(item["proof"])) <= MAX_QWEN_TOKENS
    ]
    print(f"Filtered out {len(processed_data) - len(filtered_processed_data)} proofs due to length.")

    full_output_path = Path(__file__).parent / output_path
    with open(full_output_path, "w") as f:
        for item in filtered_processed_data:
            f.write(json.dumps(item) + "\n")

    print(f"Statistics for {output_path}:")
    print(f"- {len([item for item in filtered_processed_data])} ProofBench proofs")
    print(
        f"- Correct Proofs: {len([item for item in filtered_processed_data if item['expected_judgement'] == JUDGEMENT_YES])}"
    )
    print(
        f"- Incorrect Proofs: {len([item for item in filtered_processed_data if item['expected_judgement'] == JUDGEMENT_NO])}"
    )
    print("-" * 20)


def prepare_bon_binary_data(output_path):
    print("Preparing Selection data.")
    processed_data = load_hf_data(split="best_of_n")
    filtered_processed_data = [
        item for item in processed_data if len(qwen3_tokenizer.encode(item["proof"])) <= MAX_QWEN_TOKENS
    ]
    final_data = []

    # Group by problem_id and model_id and keep proofs in a list with their expert ratings
    grouped_data = defaultdict(list)
    for item in filtered_processed_data:
        key = (item["problem_id"], item["metadata"]["model_id"])
        grouped_data[key].append(item)
    for key, items in grouped_data.items():
        problem_id, model_id = key
        proofs = [item["proof"] for item in items]
        expert_ratings = [item["metadata"]["expert_rating"] for item in items]
        max_score = max(expert_ratings)
        best_proof_indices = [i for i, score in enumerate(expert_ratings) if score == max_score]
        final_data.append(
            {
                "problem_id": problem_id,
                "model_id": model_id,
                "problem": items[0]["problem"],
                "proofs": proofs,
                "expert_ratings": expert_ratings,
                "best_proof_indices": best_proof_indices,
                "rubric": items[0]["rubric"],
                "ground_truth_proof": items[0]["ground_truth_proof"],
                "metadata": {
                    "model_id": model_id,
                },
            }
        )
    with open(output_path, "w") as f:
        for item in final_data:
            f.write(json.dumps(item) + "\n")
    print(f"Saved {len(final_data)} problems to {output_path}")


def load_hf_data(split: str):
    ds = datasets.load_dataset("wenjiema02/ProofBench")[split]
    result = []
    for x in ds:
        result.append(
            {
                "problem_id": x["problem_id"],
                "problem": x["problem"],
                "proof": x["model_solution"],
                "rubric": x["marking_scheme"],
                "ground_truth_proof": x["reference_solution"],
                "expected_judgement": JUDGEMENT_YES if x["expert_rating"] >= 6 else JUDGEMENT_NO,
                "metadata": {
                    "model_id": x["generator"],
                    "ground_truth_proof": x["reference_solution"],
                    "expert_rating": x["expert_rating"],
                },
            }
        )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="test.jsonl", help="Path to save main dataset")
    parser.add_argument(
        "--bon_output_path", type=str, default="bon_test.jsonl", help="Path to save BON binary dataset"
    )
    args = parser.parse_args()

    prepare_verification_data(args.output_path)
    prepare_bon_binary_data(args.bon_output_path)
