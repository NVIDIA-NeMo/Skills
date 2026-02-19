# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
from typing import Any

from math_verify import StringExtractionConfig, parse, verify

from nemo_skills.evaluation.evaluator.math import MathEvaluator
from nemo_skills.evaluation.math_grader import math_equal
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def relaxed_equal(gt_answer: Any, predicted_answer: Any) -> bool:
    """
    Relaxed equality check with:
    1. Case-insensitive MCQ matching
    2. Dict/list comparison using math_equal recursively
    """
    if predicted_answer is None:
        return gt_answer is None

    try:
        predicted_answer = json.loads(predicted_answer)
    except Exception:
        pass  # keep original string form
    try:
        gt_answer = json.loads(gt_answer)
    except Exception:
        pass  # keep original string form

    if isinstance(predicted_answer, dict):
        if not isinstance(gt_answer, dict):
            # check if any of the values in predicted_answer are equal to gt_answer
            return any(relaxed_equal(gt_answer, p) for p in predicted_answer.values())

        # check if all the keys in gt_answer are in predicted_answer and if the values are equal; ok for predicted_answer to have more keys
        return all(
            k in predicted_answer and relaxed_equal(gt_answer[k], predicted_answer[k]) for k in gt_answer.keys()
        )

    if isinstance(predicted_answer, list):
        if not isinstance(gt_answer, list):
            # check if any of the values in predicted_answer are equal to gt_answer
            return any(relaxed_equal(gt_answer, p) for p in predicted_answer)
        # check if the lengths are equal and if all the values are equal
        return len(gt_answer) == len(predicted_answer) and all(
            relaxed_equal(e, p) for e, p in zip(gt_answer, predicted_answer)
        )

    # Try case-insensitive MCQ matching
    # TODO: add support for numeric and roman numeral MCQs (i.e. "1", "I", "2", "II", etc.)
    mcq_options = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    norm_gt_mcq = str(gt_answer).strip().upper()
    norm_pred_mcq = str(predicted_answer).strip().upper()
    is_mcq = re.fullmatch("|".join(mcq_options), norm_gt_mcq)
    if is_mcq:
        parsed_gt = parse(norm_gt_mcq, [StringExtractionConfig(strings=tuple(mcq_options))])
        parsed_pred = parse(norm_pred_mcq, [StringExtractionConfig(strings=tuple(mcq_options))])
        mcq_result = verify(parsed_gt, parsed_pred)
        if mcq_result:
            return mcq_result

    return math_equal(str(gt_answer), str(predicted_answer))


class DSBenchEvaluator(MathEvaluator):
    def __init__(self, config: dict, num_parallel_requests=10):
        super().__init__(config, num_parallel_requests)
        self.eval_config.extract_regex = r"(?:The final answer is |\\boxed=)(.+)$"

    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        """Evaluate single DSBench problem with relaxed fallback."""
        # First try standard math evaluation
        data_point = await super().eval_single(data_point)

        # If symbolic_correct is False, try relaxed_equal
        if not data_point["symbolic_correct"]:
            expected_answer = data_point["expected_answer"]
            predicted_answer = data_point["predicted_answer"]

            if relaxed_equal(expected_answer, predicted_answer):
                data_point["symbolic_correct"] = True

        return data_point
