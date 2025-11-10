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

"""Extract predicted answers from generation outputs and optionally perform majority voting."""

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List

from nemo_skills.evaluation.math_grader import extract_answer

LOG = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract predicted_answer, and optionally fill majority voting.")
    parser.add_argument("--input_dir", required=True, help="Directory with generation output files (expects .jsonl)")
    parser.add_argument("--output_dir", required=True, help="Directory where processed files will be written")
    parser.add_argument(
        "--predicted_answer_regex",
        default=None,
        help="Optional regular expression to extract the predicted answer. If provided, the first capturing group is used.",
    )
    parser.add_argument(
        "--predicted_answer_regex_field",
        default=None,
        help="Optional field in the input file to extract the predicted answer regex. If provided, will use the field to extract the predicted answer regex.",
    )
    parser.add_argument(
        "--majority_voting",
        action="store_true",
        help="If set, compute majority voting per problem to fill expected_answer and majority_voting_rate.",
    )
    return parser.parse_args()


def collect_predictions(
    input_dir: Path,
    predicted_answer_regex: str | None,
    predicted_answer_regex_field: str | None,
) -> tuple[Dict[Path, List[dict]], DefaultDict[str, Counter], DefaultDict[str, int]]:
    """Load generation files, extract predictions, and tally answers per problem.

    Returns a tuple containing:
      * mapping of input file paths to their enriched samples,
      * per-problem counters of predicted answers,
      * per-problem totals of considered answers.
    """

    file_samples: Dict[Path, List[dict]] = {}
    answer_counts: DefaultDict[str, Counter] = defaultdict(Counter)
    totals: DefaultDict[str, int] = defaultdict(int)

    for file_path in input_dir.glob("*.jsonl"):
        samples: List[dict] = []
        with open(file_path) as fin:
            for line in fin:
                sample = json.loads(line)
                predicted_answer = extract_answer(
                    sample["generation"],
                    extract_from_boxed=False if predicted_answer_regex or predicted_answer_regex_field else True,
                    extract_regex=predicted_answer_regex
                    if predicted_answer_regex
                    else sample["metadata"][predicted_answer_regex_field],
                )
                sample["predicted_answer"] = predicted_answer
                samples.append(sample)

                if predicted_answer:
                    answer_counts[sample["problem"]][predicted_answer] += 1
                    totals[sample["problem"]] += 1

        file_samples[file_path] = samples

    return file_samples, answer_counts, totals


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.predicted_answer_regex and args.predicted_answer_regex_field:
        raise ValueError("Only one of --predicted_answer_regex or --predicted_answer_regex_field can be provided.")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    file_samples, answer_counts, totals = collect_predictions(
        input_dir, args.predicted_answer_regex, args.predicted_answer_regex_field
    )
    if not file_samples:
        LOG.warning("No .jsonl files found inside %s", input_dir)
        return

    for file_path, samples in file_samples.items():
        destination = output_dir / file_path.name
        destination.parent.mkdir(parents=True, exist_ok=True)

        with open(destination, "w") as fout:
            for sample in samples:
                if args.majority_voting or "expected_answer" not in sample:
                    majority_answer = ""
                    counter = answer_counts.get(sample["problem"], Counter())
                    if counter:
                        majority_answer, majority_votes = counter.most_common(1)[0]
                    sample["expected_answer"] = majority_answer
                    total_votes = totals[sample["problem"]]
                    sample["majority_voting_agreement_rate"] = (
                        f"{majority_votes}/{total_votes}" if total_votes else "0/0"
                    )
                    sample["majority_voting_agreement_at_n"] = (
                        round(majority_votes / total_votes, 6) if total_votes else 0.0
                    )
                fout.write(json.dumps(sample) + "\n")

        LOG.info("Wrote %s samples to %s", len(samples), destination)


if __name__ == "__main__":
    main()
