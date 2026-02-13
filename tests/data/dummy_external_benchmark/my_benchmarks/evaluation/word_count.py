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

import re

from nemo_skills.evaluation.evaluator.base import BaseEvaluator


class WordCountEvaluator(BaseEvaluator):
    async def eval_single(self, data_point):
        """Extract predicted answer and compare to expected."""
        match = re.search(r"\\boxed\{(\d+)\}", data_point["generation"])
        predicted = int(match.group(1)) if match else None

        return {
            "predicted_answer": predicted,
            "is_correct": predicted == data_point["expected_answer"],
        }
