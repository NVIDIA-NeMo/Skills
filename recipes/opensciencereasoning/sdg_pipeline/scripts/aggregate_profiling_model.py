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
import glob
import json
from collections import defaultdict
from pathlib import Path

from nemo_skills.evaluation.metrics.utils import is_correct_judgement
from recipes.opensciencereasoning.sdg_pipeline.scripts.utils.constants import BASE_FIELDS


def main():
    """Per-model profiling aggregation.

    Reads judged outputs, computes pass_rate and pass_at_n per problem,
    and writes one record per problem with BASE_FIELDS plus a profiling_entry dict:
      {"model": <model_name>, "pass_rate": <float>, "pass_at_n": "<str>"}
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgement_dir", required=True, help="Directory with judgement output-rs*.jsonl files")
    parser.add_argument("--output_file", required=True, help="Where to write updated result.jsonl")
    parser.add_argument("--model_name", required=True, help="Model name to record in the profiling entry")
    args = parser.parse_args()

    # Stats are keyed by (id, problem) where available — the upstream RL corpus
    # guarantees uniqueness on that pair. When neither id nor problem exists,
    # fall back to (line_number, None): `_line_number` is injected into each
    # sample below (one counter per input file) so seeds still align by row.
    def sample_key(sample):
        sid = sample.get("id")
        problem = sample.get("problem")
        if sid is not None or problem is not None:
            return (sid, problem)
        return ("_lineno", sample["_line_number"])

    # Memory-efficient streaming aggregation: at 1M+ problems × 5 seeds with
    # multi-KB `generation` / `judgement` text per row, keeping full rows in
    # memory would use tens of GB. Instead, keep only BASE_FIELDS for
    # emission and a small counters dict for stats.
    judgements_by_key = defaultdict(lambda: {"total": 0, "correct": 0})
    base_by_key = {}
    ordered_keys = []

    files = sorted(glob.glob(f"{args.judgement_dir}/output-rs*.jsonl"))
    if not files:
        raise FileNotFoundError(
            f"No output-rs*.jsonl files found in {args.judgement_dir}. "
            f"aggregate_profiling_model expects per-seed judged outputs."
        )
    for path in files:
        with open(path) as f:
            for line_no, line in enumerate(f):
                sample = json.loads(line)
                sample["_line_number"] = line_no
                key = sample_key(sample)
                stats = judgements_by_key[key]
                stats["total"] += 1
                if is_correct_judgement(sample["judgement"]):
                    stats["correct"] += 1
                if key not in base_by_key:
                    base_by_key[key] = {k: v for k, v in sample.items() if k in BASE_FIELDS}
                    ordered_keys.append(key)

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as fout:
        for key in ordered_keys:
            stats = judgements_by_key[key]
            total = stats["total"]
            correct = stats["correct"]
            pass_rate = correct / total if total > 0 else 0.0
            pass_at_n = f"{correct}/{total}" if total > 0 else "0/0"

            record = base_by_key[key]
            record["profiling_entry"] = {
                "model": args.model_name,
                "pass_rate": round(pass_rate, 6),
                "pass_at_n": pass_at_n,
            }
            fout.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
