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

import json
from pathlib import Path

import datasets


def process_row(row, source):
    return {
        "problem": row["problem"],
        "expected_answer": str(row["answer"]),
        "id": f"{source}-{row['problem_idx']}",
    }


def load_jsonl_problems(file_path, target_ids):
    aime24_problems = []
    hmmt24_problems = []

    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            problem_id = data.get("id")

            if problem_id not in target_ids:
                continue

            if data.get("subset_for_metrics") == "aime24":
                problem_data = {
                    "problem": data["problem"],
                    "expected_answer": data["expected_answer"],
                    "id": f"{data['id']}",
                }
                aime24_problems.append(problem_data)
            elif data.get("subset_for_metrics") == "hmmt-24-25":
                problem_data = {
                    "problem": data["problem"],
                    "expected_answer": data["expected_answer"],
                    "id": f"{data['id']}",
                }
                hmmt24_problems.append(problem_data)

    return aime24_problems, hmmt24_problems


def load_ids_from_file(file_path):
    with open(file_path, "r") as f:
        ids = [line.strip() for line in f if line.strip()]
    return set(ids)


def main():
    data_dir = Path(__file__).absolute().parent

    ids_file = data_dir / "test.txt"
    target_ids = load_ids_from_file(ids_file)
    print(f"Loaded {len(target_ids)} target IDs from {ids_file}")

    comp_math_file = data_dir.parent / "comp-math-24-25" / "test.txt"
    aime24_problems, hmmt24_problems = load_jsonl_problems(comp_math_file, target_ids)
    print(f"Found {len(aime24_problems)} AIME 2024 problems and {len(hmmt24_problems)} HMMT 2024 problems")

    cmimc_ds = datasets.load_dataset("MathArena/cmimc_2025")["train"]

    cmimc_problems = []
    for row in cmimc_ds:
        source_id = f"cmimc_2025-{row['problem_idx']}"
        if source_id in target_ids:
            problem_data = process_row(row, "cmimc_2025")
            problem_data["id"] = source_id
            cmimc_problems.append(problem_data)

    print(f"Found {len(cmimc_problems)} CMIMC 2025 problems")

    all_problems = cmimc_problems + aime24_problems + hmmt24_problems

    final_ds = datasets.Dataset.from_list(all_problems)

    final_ds = final_ds.filter(lambda x: x["expected_answer"].isdigit())

    final_ds = final_ds.sort("id")

    output_file = data_dir / "test.jsonl"
    with open(output_file, "wt") as fout:
        for row in final_ds:
            fout.write(json.dumps(row) + "\n")

    print(f"Wrote {len(final_ds)} problems to {output_file}")


if __name__ == "__main__":
    main()
