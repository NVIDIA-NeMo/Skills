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
import hashlib
import json
from pathlib import Path

import datasets


def load_jsonl(file_path):
    data = []
    full_path = Path(__file__).parent / file_path
    with open(full_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def prepare_bon_binary_data(output_path):
    dataset = datasets.load_dataset("INSAIT-Institute/OPC", revision="dcc3b4804e2d126ea34b13e3e0cd998c3302644b")
    print("Preparing Selection data.")
    # Group by problem and model_id
    grouped_data = {}
    for item in dataset["pass_at_n"]:
        problem_text = item["problem"]
        model_id = item["model_id"]
        key = (problem_text, model_id)
        grouped_data[key] = grouped_data.get(key, []) + [item]

    final_data = []
    for key, items in grouped_data.items():
        problem_text, model_id = key
        proofs = [item["solution"] for item in items]
        scores = [item["score"][0] for item in items]
        max_score = max(scores)
        best_proof_indices = [i for i, score in enumerate(scores) if score == max_score]

        final_data.append(
            {
                "problem_id": hashlib.sha256(problem_text.encode()).hexdigest(),
                "problem": problem_text,
                "model_id": model_id,
                "proofs": proofs,
                "expert_ratings": [7 * x for x in scores],
                "best_proof_indices": best_proof_indices,
            }
        )

    full_output_path = Path(__file__).parent / output_path
    with open(full_output_path, "w") as f:
        for item in final_data:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(final_data)} problems to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bon_output_path", type=str, default="bon_test.jsonl", help="Path to save BON binary dataset"
    )
    args = parser.parse_args()

    prepare_bon_binary_data(args.bon_output_path)
