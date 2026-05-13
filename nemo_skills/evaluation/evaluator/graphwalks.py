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

import json
import logging
import re

from tqdm import tqdm

from nemo_skills.evaluation.evaluator.base import BaseEvaluatorConfig
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def get_list(response: str) -> tuple[list[str], bool]:
    """Parse the predicted node list from the last non-empty line of the response.

    Expects the format: ``Final Answer: [node1, node2, ...]``

    Returns:
        (nodes, parse_failed) where parse_failed is True when the expected
        format was not found.

    Reference: https://huggingface.co/datasets/openai/graphwalks
    """
    lines = [line for line in response.strip().split("\n") if line.strip()]
    if not lines:
        return [], True

    last_line = lines[-1]
    match = re.search(r"Final Answer:\s*\[(.*)\]", last_line)
    if match:
        content = match.group(1)
        if not content.strip():
            return [], False
        return [item.strip() for item in content.split(",") if item.strip()], False

    return [], True


def eval_graphwalks(cfg):
    cfg = BaseEvaluatorConfig(**cfg)

    jsonl_file = cfg.input_file
    with open(jsonl_file, "rt", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]

    with open(jsonl_file, "wt", encoding="utf-8") as fout:
        for sample in tqdm(data):
            predicted_list, parse_failed = get_list(sample["generation"])
            predicted_nodes = set(predicted_list)

            try:
                expected_nodes = set(json.loads(sample["expected_answer"]))
            except (json.JSONDecodeError, TypeError):
                expected_nodes = set()

            if not expected_nodes and not predicted_nodes:
                f1 = 1.0
            elif not predicted_nodes or not expected_nodes:
                f1 = 0.0
            else:
                tp = len(predicted_nodes & expected_nodes)
                precision = tp / len(predicted_nodes)
                recall = tp / len(expected_nodes)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            sample["f1"] = f1
            sample["parse_failed"] = parse_failed
            fout.write(json.dumps(sample) + "\n")
