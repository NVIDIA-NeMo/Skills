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

import logging

from nemo_skills.evaluation.chemistry_grader import chemistry_equal
from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.evaluation.math_grader import extract_answer
from nemo_skills.utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class ChemistryEvaluatorConfig(BaseEvaluatorConfig):
    # if True will not attempt to re-extract based on \boxed or regex
    use_predicted_answer_key: bool = False

    extract_from_boxed: bool = True
    # only used if extract_from_boxed is False
    extract_regex: str = r"The final answer is (.+)$"
    # if True: try regex first, then boxed (regardless of extract_from_boxed)
    relaxed_extraction: bool = False

    # compare connectivity only (ignore E/Z and R/S stereochemistry)
    ignore_stereo: bool = False
    # tolerances for numeric answers (e.g. property/regression tasks)
    numeric_rel_tol: float = 1e-6
    numeric_abs_tol: float = 1e-9


class ChemistryEvaluator(BaseEvaluator):
    """Deterministic RDKit/numeric grading for chemistry answers.

    Sets ``data_point["symbolic_correct"]`` analogously to :class:`MathEvaluator`,
    so it can run inline during generation (``eval_single``) or as a batch
    re-grading pass (``eval_full``).
    """

    def __init__(self, config: dict, num_parallel_requests=10):
        super().__init__(config, num_parallel_requests)
        self.eval_config = ChemistryEvaluatorConfig(**self.config)

    async def eval_single(self, data_point: dict[str, any]) -> dict[str, any]:
        if not self.eval_config.use_predicted_answer_key:
            # Extraction is identical to math (boxed/regex); chemistry-specific
            # normalization happens inside chemistry_equal during comparison.
            data_point["predicted_answer"] = extract_answer(
                data_point["generation"],
                extract_from_boxed=self.eval_config.extract_from_boxed,
                extract_regex=self.eval_config.extract_regex,
                relaxed=self.eval_config.relaxed_extraction,
            )
        else:
            if "predicted_answer" not in data_point:
                raise ValueError(
                    "predicted_answer key not found in the data_point. Set use_predicted_answer_key=False to re-extract"
                )

        data_point["symbolic_correct"] = chemistry_equal(
            data_point["expected_answer"],
            data_point["predicted_answer"],
            metadata=data_point.get("metadata"),
            ignore_stereo=self.eval_config.ignore_stereo,
            numeric_rel_tol=self.eval_config.numeric_rel_tol,
            numeric_abs_tol=self.eval_config.numeric_abs_tol,
        )
        return data_point
