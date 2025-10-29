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

from recipes.opensciencereasoning.scripts.SDG_pipeline.constants import BASE_FIELDS


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

    # First level: list of allowed values
    first_name = names[0]
    first_allowed = topics_structure.get(first_name, [])
    first_value = sample.get(first_name)
    if first_value == "Other":
        set_undefined_from(1)
        return
    if first_value not in first_allowed:
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
            set_undefined_from(i + 1)
            return

        mapping_or_list = topics_structure.get(curr_name, {})
        if isinstance(mapping_or_list, dict):
            allowed = mapping_or_list.get(prev_value, [])
        else:
            allowed = mapping_or_list

        if curr_value not in allowed:
            sample[curr_name] = "undefined"
            set_undefined_from(i + 1)
            return


def aggregate_topics(input_files: dict, output_file: str, topics_structure: dict, names: list):
    """Merge per-level classification outputs into a single JSONL.

    - `input_files`: mapping from label key (e.g., "topics") to its output.jsonl
    - `topics_structure`: control structure used to filter invalid hierarchical pairs
    - `names`: ordered list of keys defining the hierarchy (e.g., ["topics", "subtopics"]).
    """
    data = {}
    for topic_key, file in input_files.items():
        with open(file, "r") as f:
            for line in f:
                sample = json.loads(line)
                if sample["problem"] not in data:
                    data[sample["problem"]] = sample
                data[sample["problem"]][topic_key] = sample[topic_key]
    with open(output_file, "w") as f:
        for sample in data.values():
            check_topic_structure(sample, topics_structure, names)
            sample = {key: value for key, value in sample.items() if key in BASE_FIELDS + names}
            f.write(json.dumps(sample) + "\n")


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
    aggregate_topics(args.input_files, args.output_file, args.topics_structure, args.names)


if __name__ == "__main__":
    main()
