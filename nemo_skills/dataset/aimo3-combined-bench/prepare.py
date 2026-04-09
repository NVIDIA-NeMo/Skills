# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


def normalize(problem_text):
    """Normalize problem text for deduplication."""
    return " ".join(problem_text.split()).strip()[:500]


def load_jsonl(path):
    entries = []
    with open(path) as f:
        for line in f:
            entries.append(json.loads(line))
    return entries


if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    dataset_root = data_dir.parent

    # Load source datasets (order defines priority for duplicates)
    sources = [
        ("aimo3-answerbench-50", dataset_root / "aimo3-answerbench-50" / "test.jsonl"),
        ("aimo3-intbench50", dataset_root / "aimo3-intbench50" / "test.jsonl"),
        ("apex-shortlist", dataset_root / "apex-shortlist" / "test.jsonl"),
    ]

    seen_problems = set()
    combined = []

    for source_name, path in sources:
        for entry in load_jsonl(path):
            key = normalize(entry["problem"])
            if key in seen_problems:
                continue
            seen_problems.add(key)
            entry["source_dataset"] = source_name
            combined.append(entry)

    # Load verified 2025 problems from AoPS candidates
    aops_candidates_path = data_dir.parent.parent.parent / "aops_hard_candidates_2025_2026.jsonl"
    if aops_candidates_path.exists():
        for entry in load_jsonl(aops_candidates_path):
            if not entry.get("expected_answer"):
                continue
            key = normalize(entry["problem"])
            if key in seen_problems:
                continue
            seen_problems.add(key)
            combined.append(
                {
                    "problem": entry["problem"],
                    "expected_answer": entry["expected_answer"],
                    "contest": entry.get("contest", ""),
                    "year": entry.get("year", ""),
                    "problem_id": entry.get("problem_id", ""),
                    "source_dataset": "aops-2025-verified",
                    "source_url": entry.get("source_url", ""),
                }
            )

    output_file = data_dir / "test.jsonl"
    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in combined:
            json.dump(entry, fout, ensure_ascii=False)
            fout.write("\n")

    print(f"Written {len(combined)} unique problems to {output_file}")
