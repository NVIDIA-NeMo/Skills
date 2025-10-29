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

"""Aggregate solution generations into a single JSONL file."""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

from nemo_skills.evaluation.metrics.utils import is_correct_judgement
from recipes.opensciencereasoning.scripts.SDG_pipeline.constants import BASE_FIELDS

LOG = logging.getLogger(__name__)


DEFAULT_OUTPUT_PATTERN = "output*.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate solution generations into final_result.jsonl")
    parser.add_argument("--input_dir", required=True, help="Directory containing generation or judgement outputs")
    parser.add_argument("--output_file", required=True, help="Where to write the aggregated JSONL")
    parser.add_argument("--generation_model", required=True, help="Identifier of the generation model to record")

    return parser.parse_args()


def aggregate_samples(files: Iterable[Path]) -> List[Dict]:
    """Read generation/judgement files and enrich them with correctness metrics."""
    per_problem_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    all_samples: List[Dict] = []

    for file_path in files:
        LOG.info("Reading %s", file_path)
        with open(file_path) as fin:
            for line in fin:
                sample = json.loads(line)

                sample = {
                    key: value
                    for key, value in sample.items()
                    if key
                    in BASE_FIELDS
                    + [
                        "predicted_answer",
                        "generation",
                        "judgement",
                        "majority_voting_agreement_rate",
                        "majority_voting_agreement_at_n",
                    ]
                }

                if "judgement" in sample:
                    is_correct = is_correct_judgement(sample["judgement"])
                else:
                    is_correct = sample["predicted_answer"] == sample["expected_answer"]

                sample["is_correct"] = is_correct

                per_problem_stats[sample["problem"]]["total"] += 1
                if is_correct:
                    per_problem_stats[sample["problem"]]["correct"] += 1

                all_samples.append(sample)

    for sample in all_samples:
        stats = per_problem_stats[sample["problem"]]
        total = stats["total"]
        correct = stats["correct"]
        sample["generation_model_pass_rate"] = round(correct / total, 6) if total else 0.0
        sample["generation_model_pass_at_n"] = f"{correct}/{total}" if total else "0/0"

    return all_samples


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = sorted(input_dir.glob(DEFAULT_OUTPUT_PATTERN))
    if not files:
        LOG.warning("No files matched %s in %s", DEFAULT_OUTPUT_PATTERN, input_dir)
        open(args.output_file, "w").close()
        return

    samples = aggregate_samples(files)

    with open(args.output_file, "w") as fout:
        for sample in samples:
            sample["generation_model"] = args.generation_model
            fout.write(json.dumps(sample) + "\n")

    LOG.info("Wrote %s samples to %s", len(samples), args.output_file)


if __name__ == "__main__":
    main()
