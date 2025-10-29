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

from nemo_skills.evaluation.metrics.utils import is_correct_judgement
from recipes.opensciencereasoning.scripts.SDG_pipeline.constants import BASE_FIELDS


def main():
    """Postprocess judged generations to add difficulty_model, difficulty_model_pass_rate, difficulty_model_pass_at_n.

    This script expects judged outputs. It aggregates Yes/No judgements per problem across
    seeds and writes enriched samples with:
      - difficulty_model: model used for generations
      - difficulty_model_pass_rate: decimal ratio correct/total (e.g., 0.5)
      - difficulty_model_pass_at_n: string fraction "correct/total" (e.g., 2/4)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgement_dir", required=True, help="Directory with judgement output-rs*.jsonl files")
    parser.add_argument("--output_file", required=True, help="Where to write updated final_result.jsonl")
    parser.add_argument("--difficulty_model_pass_rate", required=True, help="Model used for generations to record")
    args = parser.parse_args()

    # Aggregate judgements per id across random seeds
    judgements_by_problem = defaultdict(lambda: {"total": 0, "correct": 0})
    samples = []

    files = sorted(glob.glob(f"{args.judgement_dir}/output*.jsonl"))
    for i, path in enumerate(files):
        with open(path) as f:
            for line in f:
                sample = json.loads(line)
                if i == 0:
                    samples.append(sample)
                judgements_by_problem[sample["problem"]]["total"] += 1
                judgements_by_problem[sample["problem"]]["correct"] += (
                    1 if is_correct_judgement(sample["judgement"]) else 0
                )

    # Write updated records with required keys
    with open(args.output_file, "w") as fout:
        for sample in samples:
            stats = judgements_by_problem[sample["problem"]]
            total = stats["total"]
            correct = stats["correct"]
            # pass_rate as decimal; pass_at_n as fraction string
            pass_rate = correct / total if total > 0 else 0.0
            pass_at_n = f"{correct}/{total}" if total > 0 else "0/0"

            sample = {key: value for key, value in sample.items() if key in BASE_FIELDS}
            sample["difficulty_model"] = args.difficulty_model_pass_rate
            sample["difficulty_model_pass_rate"] = round(pass_rate, 6)
            sample["difficulty_model_pass_at_n"] = pass_at_n
            fout.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    main()
