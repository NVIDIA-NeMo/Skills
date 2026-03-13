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
from pathlib import Path


NOT_AVAILABLE = "Not available for this problem"


def format_entry(raw_entry):
    subset = raw_entry["subset_for_metrics"]

    entry = {
        "problem": raw_entry["problem"],
        "problem_id": None,
        "ground_truth_proof": NOT_AVAILABLE,
        "subset_for_metrics": subset,
    }

    if subset == "imoproofbench":
        entry["problem_id"] = raw_entry["problem_id"]
        entry["ground_truth_proof"] = raw_entry["reference_solution"]
        entry["metadata"] = {"category": raw_entry["category"], "level": raw_entry["level"], "source": raw_entry["source"]}
    elif subset == "putnam2025":
        entry["problem_id"] = raw_entry["problem_name"]
        entry["ground_truth_proof"] = raw_entry.get("informal_solution", NOT_AVAILABLE)
        entry["metadata"] = {}
    elif subset == "olympiads2025":
        entry["problem_id"] = raw_entry["problem_id"]
        entry["metadata"] = raw_entry["metadata"]

    return entry


if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    input_file = data_dir / "data" / "problems.txt"
    output_file = data_dir / "test.jsonl"

    with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line in fin:
            raw_entry = json.loads(line.strip())
            entry = format_entry(raw_entry)
            fout.write(json.dumps(entry) + "\n")

    print(f"Wrote {output_file}")
