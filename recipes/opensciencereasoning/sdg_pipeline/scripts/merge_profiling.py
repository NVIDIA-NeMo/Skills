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

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from recipes.opensciencereasoning.sdg_pipeline.scripts.utils.constants import BASE_FIELDS


def main():
    """Merge per-model profiling results into a single profiling array per problem.

    Reads N per-model result files (each containing a profiling_entry dict per record)
    and combines them into:
      BASE_FIELDS + "profiling": [
          {"model": "ModelA", "pass_rate": 0.5, "pass_at_n": "2/4"},
          {"model": "ModelB", "pass_rate": 0.8, "pass_at_n": "4/5"},
      ]
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_result_files", required=True, type=json.loads, help="JSON list of per-model result file paths"
    )
    parser.add_argument("--output_file", required=True, help="Where to write the merged final_result.jsonl")
    args = parser.parse_args()

    # Join key: (id, problem) where available — the upstream RL corpus
    # guarantees uniqueness on that pair. When neither id nor problem exists,
    # fall back to (_lineno, line_number) so per-model files still align by row.
    def record_key(record):
        sid = record.get("id")
        problem = record.get("problem")
        if sid is not None or problem is not None:
            return (sid, problem)
        return ("_lineno", record["_line_number"])

    profiling_by_key = defaultdict(list)
    base_records_by_key = {}
    per_file_key_sets = []

    for path in args.model_result_files:
        keys_in_file = set()
        with open(path) as f:
            for line_no, line in enumerate(f):
                record = json.loads(line)
                record["_line_number"] = line_no
                key = record_key(record)
                keys_in_file.add(key)
                profiling_by_key[key].append(record["profiling_entry"])
                if key not in base_records_by_key:
                    base_records_by_key[key] = {k: v for k, v in record.items() if k in BASE_FIELDS}
        per_file_key_sets.append(keys_in_file)

    # Assert every per-model file covers the same (id, problem) key set
    baseline = per_file_key_sets[0]
    for i, key_set in enumerate(per_file_key_sets[1:], start=1):
        missing = baseline - key_set
        extra = key_set - baseline
        if missing or extra:
            raise ValueError(
                f"Per-model file {args.model_result_files[i]} has a different (id, problem) "
                f"key set than {args.model_result_files[0]}: {len(missing)} missing, {len(extra)} extra. "
                f"Row-alignment across models is required for merge_profiling."
            )

    # Use the first model's file to determine record order
    ordered_keys = []
    seen = set()
    with open(args.model_result_files[0]) as f:
        for line_no, line in enumerate(f):
            record = json.loads(line)
            record["_line_number"] = line_no
            key = record_key(record)
            if key not in seen:
                ordered_keys.append(key)
                seen.add(key)

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as fout:
        for key in ordered_keys:
            record = base_records_by_key[key]
            record["profiling"] = profiling_by_key[key]
            fout.write(json.dumps(record) + "\n")

    # With the merged final_result.jsonl now on disk, the per-model result.jsonl
    # files are redundant intermediates. Remove the files (not the folders —
    # generation/, judgement/, logs/ stay for debugging / provenance).
    for path in args.model_result_files:
        try:
            os.remove(path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
