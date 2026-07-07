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
import logging
import os
from pathlib import Path

from recipes.opensciencereasoning.sdg_pipeline.scripts.utils.constants import BASE_FIELDS

logger = logging.getLogger(__name__)


def check_topic_structure(sample: dict, topics_structure: dict, names: list):
    """Stepwise validate and normalize hierarchical labels in `sample`.

    Rules:
      - At each level, if the predicted value is "Other", keep it and set all
        subsequent levels to "undefined".
      - If the predicted value is not in the allowed set for that level, set
        the current and all subsequent levels to "undefined".
      - Otherwise, keep the value and continue to the next level.

    This function mutates `sample` in place and always returns True, allowing
    the caller to write out the normalized sample.
    """
    if not names:
        return

    def set_undefined_from(index: int):
        for j in range(index, len(names)):
            sample[names[j]] = "undefined"
        logger.debug(f"Set {names[index:]} to 'undefined' for problem starting with: {sample['problem'][:50]}...")

    # First level: list of allowed values
    first_name = names[0]
    first_allowed = topics_structure.get(first_name, [])
    first_value = sample.get(first_name)
    if first_value == "Other":
        logger.debug(f"First level '{first_name}' is 'Other', setting subsequent levels to 'undefined'")
        set_undefined_from(1)
        return
    if first_value not in first_allowed:
        logger.warning(
            f"Invalid first level value '{first_value}' for '{first_name}', expected one of {first_allowed}"
        )
        sample[first_name] = "undefined"
        set_undefined_from(1)
        return

    # Subsequent levels: mapping from previous selection to allowed values
    for i in range(1, len(names)):
        prev_name = names[i - 1]
        curr_name = names[i]
        prev_value = sample.get(prev_name)
        curr_value = sample.get(curr_name)

        if curr_value == "Other":
            logger.debug(f"Level '{curr_name}' is 'Other', setting subsequent levels to 'undefined'")
            set_undefined_from(i + 1)
            return

        mapping_or_list = topics_structure.get(curr_name, {})
        if isinstance(mapping_or_list, dict):
            allowed = mapping_or_list.get(prev_value, [])
        else:
            allowed = mapping_or_list

        if curr_value not in allowed:
            logger.warning(
                f"Invalid value '{curr_value}' for '{curr_name}' given parent '{prev_name}={prev_value}', "
                f"expected one of {allowed}"
            )
            sample[curr_name] = "undefined"
            set_undefined_from(i + 1)
            return


def aggregate_topics(input_files: dict, output_file: str, topics_structure: dict, names: list):
    """Merge per-level classification outputs into a single JSONL.

    - `input_files`: mapping from label key (e.g., "topics") to its output.jsonl
    - `topics_structure`: control structure used to filter invalid hierarchical pairs
    - `names`: ordered list of keys defining the hierarchy (e.g., ["topics", "subtopics"]).
    """
    logger.info(f"Starting topic aggregation for {len(input_files)} topic levels: {list(input_files.keys())}")
    logger.info(f"Hierarchy order: {names}")

    data = {}
    for topic_key, file in input_files.items():
        logger.info(f"Reading {topic_key} labels from: {file}")
        if not os.path.exists(file):
            logger.warning(f"File not found for {topic_key}: {file}, samples will get 'undefined'")
            continue
        line_count = 0
        with open(file, "r") as f:
            for line in f:
                sample = json.loads(line)
                if sample["problem"] not in data:
                    data[sample["problem"]] = sample
                data[sample["problem"]][topic_key] = sample[topic_key]
                line_count += 1
        logger.info(f"Read {line_count} samples for {topic_key}")

    logger.info(f"Total unique problems: {len(data)}")
    logger.info(f"Validating topic structure and writing to: {output_file}")

    validation_stats = {"valid": 0, "modified": 0}
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for sample in data.values():
            original_values = {name: sample.get(name) for name in names}
            check_topic_structure(sample, topics_structure, names)
            modified_values = {name: sample.get(name) for name in names}

            if original_values != modified_values:
                validation_stats["modified"] += 1
            else:
                validation_stats["valid"] += 1

            sample = {key: value for key, value in sample.items() if key in BASE_FIELDS + names}
            f.write(json.dumps(sample) + "\n")

    logger.info(f"Aggregation complete: {validation_stats['valid']} valid, {validation_stats['modified']} modified")
    logger.info(f"Output written to: {output_file}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Aggregate per-level topic labeling outputs into a single JSONL.")
    parser.add_argument(
        "--input_files",
        required=True,
        type=json.loads,
        help="JSON: mapping from label key (e.g., 'topics') to its output.jsonl path",
    )
    parser.add_argument(
        "--output_file", required=True, type=str, help="Path to write aggregated JSONL after structure validation"
    )
    parser.add_argument(
        "--topics_structure",
        default=None,
        type=json.loads,
        help="JSON: allowed labels per level; dict-of-lists/dicts controlling valid pairs",
    )
    parser.add_argument(
        "--names",
        default=None,
        type=json.loads,
        help="JSON: ordered list of hierarchy keys (e.g., ['topics','subtopics'])",
    )
    args = parser.parse_args()

    logger.info("Starting topic aggregation script")
    logger.info(f"Input files: {args.input_files}")
    logger.info(f"Output file: {args.output_file}")
    logger.info(f"Names: {args.names}")

    aggregate_topics(args.input_files, args.output_file, args.topics_structure, args.names)

    logger.info("Topic aggregation script completed successfully")


if __name__ == "__main__":
    main()
