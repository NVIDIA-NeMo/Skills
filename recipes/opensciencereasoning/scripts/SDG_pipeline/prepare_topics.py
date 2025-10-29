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
from typing import List, Optional, Union


def prepare_examples(prompt_examples: dict, generation_key: str):
    """Render few-shot examples into a markdown block for the prompt.

    The input is a mapping of label → example problem text. The output
    is a string that the prompt template will interpolate into {prompt_examples}.
    """
    return (
        "---\n"
        + "---\n".join(
            [
                f"**Example {i + 1}:**\n\n**Problem:**\n{problem}\n\n**{generation_key.capitalize()}:**\n{topic}\n"
                for i, (topic, problem) in enumerate(prompt_examples.items())
            ]
        )
        + "---"
    )


def prepare_topics(
    input_file: str,
    output_file: str,
    topics_to_choose: Union[dict, List[str]],
    prompt_examples: dict,
    topic_key: Optional[str] = None,
    generation_key: str = "topic",
):
    """Prepare data for topic or subtopic labeling.

    - If topic_key is None: write each input sample with a flat list of topics.
    - If topic_key is set: only forward samples whose selected topic has subtopics,
      and attach those subtopics as the choices for the next round.

    The resulting JSONL contains fields used by the prompt template:
      - topics_to_choose: backtick-quoted, comma-separated labels
      - prompt_examples: rendered few-shot examples (empty if none)
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        for line in fin:
            sample = json.loads(line)
            # Determine allowed topics/subtopics for this sample
            if topic_key:
                prev_value = sample.get(topic_key)
                topics = topics_to_choose.get(prev_value, [])
                # Skip if there are no subtopics for this previously selected topic
                if not topics:
                    continue
                examples_source = prompt_examples.get(prev_value, {})
            else:
                topics = topics_to_choose
                examples_source = prompt_examples

            sample["topics_to_choose"] = ", ".join([f"`{topic}`" for topic in topics])
            sample["prompt_examples"] = prepare_examples(examples_source, generation_key) if examples_source else ""
            fout.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", required=True, type=str, help="Path to input JSONL with problems (and prior labels if any)"
    )
    parser.add_argument(
        "--output_file", required=True, type=str, help="Path to write JSONL prepared for the next labeling round"
    )
    parser.add_argument(
        "--topics_to_choose",
        required=True,
        type=json.loads,
        help="JSON: flat list of labels or dict mapping previous label → list of labels",
    )
    parser.add_argument(
        "--prompt_examples",
        required=True,
        type=json.loads,
        help="JSON: few-shot examples; flat mapping or mapping keyed by previous label",
    )
    parser.add_argument(
        "--topic_key",
        default=None,
        type=str,
        help="Name of the prior label key (e.g., 'topics') used to select next-level choices",
    )
    parser.add_argument(
        "--generation_key",
        default="topic",
        type=str,
        help="Name of the label key to generate in this round (e.g., 'subtopics')",
    )
    args = parser.parse_args()
    prepare_topics(
        args.input_file,
        args.output_file,
        args.topics_to_choose,
        args.prompt_examples,
        args.topic_key,
        args.generation_key,
    )
