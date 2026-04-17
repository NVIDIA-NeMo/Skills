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
import hashlib
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Union

from recipes.opensciencereasoning.sdg_pipeline.scripts.utils.constants import KEY_FIELDS

LOG = logging.getLogger(__name__)


DEFAULT_OUTPUT_PATTERN = "output*.jsonl"


def is_correct_judgement(judgement, return_none=False) -> Union[bool, None]:
    logging.info("Judgement string: %s", judgement)

    match = re.search(r"\*{0,2}Judgement\*{0,2}\s*:", judgement, re.IGNORECASE)
    if match:
        logging.info("Found 'Judgement:' in string, extracting verdict.")
        verdict = judgement[match.end() :].strip().lstrip("*").strip()
        if verdict.lower().startswith("yes"):
            return True
        elif verdict.lower().startswith("no"):
            return False

    judgement = judgement.lower().strip()
    logging.info("Normalized judgement string: %s", judgement)
    if judgement == "yes":
        return True
    if judgement == "no":
        return False
    if return_none:
        return None
    else:
        return False  # improper judgement format, so have to judge as false


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate solution generations into final_result.jsonl")
    parser.add_argument("--generation_dir", required=True, help="Directory containing generation outputs")
    parser.add_argument(
        "--judgement_dir",
        default=None,
        help="Optional directory containing judgement outputs. When provided, judgements are mapped onto generation data using generation_id",
    )
    parser.add_argument("--output_file", required=True, help="Where to write the aggregated JSONL")
    parser.add_argument("--generation_model", required=True, help="Identifier of the generation model to record")

    return parser.parse_args()


def aggregate_samples(generation_files: Iterable[Path], judgement_files: Iterable[Path] = None) -> List[Dict]:
    """Read generation/judgement files and enrich them with correctness metrics.

    If judgement_files is provided, this function will:
    1. Load all generation data indexed by generation_id
    2. Load judgement files and map judgements to generation data by generation_id
    3. Take ONLY the judgement field from judgement files, keeping all other data from generation files
    """
    per_problem_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    all_samples: List[Dict] = []

    source_files = judgement_files if judgement_files else generation_files

    # If judgement_files are provided, load generation data first
    generation_data_map = {}
    if judgement_files:
        LOG.info("Loading generation data files for mapping...")
        for gen_file in generation_files:
            LOG.info("Reading generation data from %s", gen_file)
            with open(gen_file) as fin:
                for line in fin:
                    gen_sample = json.loads(line)
                    if not gen_sample:
                        continue
                    gen_id = gen_sample.get("generation_id")
                    if gen_id:
                        generation_data_map[gen_id] = gen_sample
        LOG.info("Loaded %d generation samples", len(generation_data_map))

    for file_path in source_files:
        LOG.info("Reading %s", file_path)
        with open(file_path) as fin:
            for line in fin:
                sample = json.loads(line)
                if not sample:
                    logging.warning("Skipping empty line in %s", file_path)
                    continue

                # If we have generation data, map judgement to it
                if judgement_files:
                    gen_id = sample.get("generation_id")
                    if not gen_id:
                        # Try to reconstruct generation_id from file and sample
                        seed_match = re.search(r"output-rs(\d+)", file_path.stem)
                        random_seed = seed_match.group(1) if seed_match else "unknown"
                        file_hash = hashlib.md5(file_path.name.encode()).hexdigest()[:8]
                        problem_id = sample.get("id", "unknown")
                        gen_id = f"{problem_id}__rs{random_seed}__{file_hash}"

                    if gen_id in generation_data_map:
                        # Use generation data as base and only take judgement from judged file
                        base_sample = generation_data_map[gen_id].copy()
                        if "judgement" in sample:
                            base_sample["judgement"] = sample["judgement"]
                        sample = base_sample
                    else:
                        LOG.warning("Generation data not found for generation_id: %s", gen_id)

                sample = {
                    key: value
                    for key, value in sample.items()
                    if key
                    in KEY_FIELDS
                    + [
                        "generation_id",
                        "predicted_answer",
                        "generation",
                        "judgement",
                        "majority_voting_agreement_rate",
                        "majority_voting_agreement_at_n",
                        "tools",
                        "conversation",
                        "num_tool_calls",
                        "serialized_output",
                        "_full_generation",
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

    generation_dir = Path(args.generation_dir)
    if not generation_dir.exists() or not generation_dir.is_dir():
        raise FileNotFoundError(f"Generation directory does not exist: {generation_dir}")

    generation_files = sorted(generation_dir.glob(DEFAULT_OUTPUT_PATTERN))
    if args.judgement_dir:
        judgement_dir = Path(args.judgement_dir)
        if not judgement_dir.exists() or not judgement_dir.is_dir():
            raise FileNotFoundError(f"Judgement directory does not exist: {judgement_dir}")

        judgement_files = sorted(judgement_dir.glob(DEFAULT_OUTPUT_PATTERN))

        if not judgement_files:
            LOG.warning("No files matched %s in %s", DEFAULT_OUTPUT_PATTERN, judgement_dir)
            Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
            open(args.output_file, "w").close()
            return
        if not generation_files:
            LOG.warning("No generation data files matched %s in %s", DEFAULT_OUTPUT_PATTERN, generation_dir)
        samples = aggregate_samples(generation_files, judgement_files)
    else:
        if not generation_files:
            LOG.warning("No files matched %s in %s", DEFAULT_OUTPUT_PATTERN, generation_dir)
            Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
            open(args.output_file, "w").close()
            return
        samples = aggregate_samples(generation_files)

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w") as fout:
        for sample in samples:
            sample["generation_model"] = args.generation_model
            fout.write(json.dumps(sample) + "\n")

    LOG.info("Wrote %s samples to %s", len(samples), args.output_file)


if __name__ == "__main__":
    main()
