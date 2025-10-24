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
from difflib import SequenceMatcher

from tqdm import tqdm

from nemo_skills.evaluation.evaluator.base import BaseEvaluatorConfig
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def eval_mrcr(cfg):
    cfg = BaseEvaluatorConfig(**cfg)

    def grade(response, answer, random_string_to_prepend) -> float:
        """
        Compare response and answer.
        # Offical grading function: https://huggingface.co/datasets/openai/mrcr
        """
        if not response.startswith(random_string_to_prepend):
            return 0
        response = response.removeprefix(random_string_to_prepend)
        answer = answer.removeprefix(random_string_to_prepend)
        return float(SequenceMatcher(None, response, answer).ratio())

    jsonl_file = cfg.input_file
    with open(jsonl_file, "rt", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]
    with open(jsonl_file, "wt", encoding="utf-8") as fout:
        for sample in tqdm(data):
            sample["seq_match_ratio"] = grade(
                sample["generation"], sample["expected_answer"], sample["random_string_to_prepend"]
            )
            fout.write(json.dumps(sample) + "\n")
