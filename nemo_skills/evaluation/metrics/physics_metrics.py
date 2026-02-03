# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import re
from typing import Union

from nemo_skills.evaluation.metrics.math_metrics import MathMetrics
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


def is_correct_physics_judgement(judgement, return_none=False) -> Union[bool, None]:
    """Parse physics judgement that returns [Correct] or [Incorrect]."""
    if judgement:
        # Look for [Correct] or [Incorrect] patterns (case insensitive)
        if re.search(r"\[correct\]", judgement, re.IGNORECASE):
            return True
        elif re.search(r"\[incorrect\]", judgement, re.IGNORECASE):
            return False

    if return_none:
        return None
    else:
        return False  # improper judgement format, so have to judge as false


class PhysicsMetrics(MathMetrics):
    def __init__(self, compute_no_answer: bool = False, answer_key: str = "generation"):
        super().__init__(compute_no_answer=compute_no_answer)
        self.answer_key = answer_key

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        correctness_dict = {}
        if "symbolic_correct" in prediction:
            correctness_dict["symbolic_correct"] = prediction["symbolic_correct"]
        if "judgement" in prediction:
            correctness_dict["judge_correct"] = is_correct_physics_judgement(prediction["judgement"])
        if "judge_correct" in correctness_dict and "symbolic_correct" in correctness_dict:
            correctness_dict["both_correct"] = (
                correctness_dict["symbolic_correct"] and correctness_dict["judge_correct"]
            )
            correctness_dict["any_correct"] = correctness_dict["symbolic_correct"] or correctness_dict["judge_correct"]

        return correctness_dict

    def get_incorrect_sample(self, prediction: dict) -> dict:
        prediction = prediction.copy()
        if "symbolic_correct" in prediction:
            prediction["symbolic_correct"] = False
        if "judgement" in prediction:
            prediction["judgement"] = "[Incorrect]"
        prediction[self.answer_key] = None
        return prediction
