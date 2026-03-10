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

from nemo_skills.evaluation.metrics.math_metrics import MathMetrics
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


class UGPhysicsMetrics(MathMetrics):
    def __init__(self, compute_no_answer: bool = False, answer_key: str = "generation"):
        super().__init__(compute_no_answer=compute_no_answer)
        self.answer_key = answer_key

    def is_correct_judgement(self, judgement: str, return_none: bool = False) -> bool:
        """Parse UGPhysics judgement that returns TRUE or FALSE in ## Equivalence Judgement section."""
        if judgement:
            # Look for the Equivalence Judgement section
            equiv_match = re.search(r"##\s*Equivalence\s*Judgement\s*\n\s*(TRUE|FALSE)", judgement, re.IGNORECASE)
            if equiv_match:
                return equiv_match.group(1).upper() == "TRUE"
            # Fallback: look for standalone TRUE/FALSE (case-insensitive), use last match
            true_false_matches = list(re.finditer(r"\b(TRUE|FALSE)\b", judgement, re.IGNORECASE))
            if true_false_matches:
                return true_false_matches[-1].group(1).upper() == "TRUE"

        # improper judgement format, so have to judge as false
        return None if return_none else False

    def get_incorrect_sample(self, prediction: dict) -> dict:
        prediction = prediction.copy()
        if "symbolic_correct" in prediction:
            prediction["symbolic_correct"] = False
        if "judgement" in prediction:
            prediction["judgement"] = "FALSE"
        prediction[self.answer_key] = None
        return prediction
