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

import asyncio
import logging
from dataclasses import asdict, field

from nemo_skills.code_execution.sandbox import get_sandbox
from nemo_skills.evaluation.math_grader import batch_evaluate_results
from nemo_skills.utils import get_logger_name, nested_dataclass

LOG = logging.getLogger(get_logger_name(__file__))


@nested_dataclass(kw_only=True)
class MathEvaluatorConfig:
    numeric_precision: int = 15
    timeout: int = 10
    # if True will not attempt to re-extract based on \boxed or regex
    use_predicted_answer_key: bool = False

    extract_from_boxed: bool = True
    # only used if extract_from_boxed is False
    extract_regex: str = r"The final answer is (.+)$"
    take_modulo: int | None = None  # will take modulo of the gt and predicted answers if not None


def eval_math(cfg):
    eval_config = MathEvaluatorConfig(**cfg.eval_config)

    eval_config = asdict(eval_config)
    batch_evaluate_results(
        input_files=cfg.input_files,
        **eval_config,
    )


@nested_dataclass(kw_only=True)
class LeanEvaluatorConfig:
    sandbox: dict = field(default_factory=lambda: {"sandbox_type": "local"})
    num_parallel_requests: int = 10
    timeout: float = 30.0
    final_answer_key: str = "**FINAL ANSWER**"
    restate_formal_statement: bool = True
    strip_theorem_from_proof: bool = True


def eval_lean4_proof(cfg):
    eval_config = LeanEvaluatorConfig(**cfg.eval_config)

    sandbox = get_sandbox(**eval_config.sandbox)
    eval_config_dict = asdict(eval_config)
    eval_config_dict.pop("sandbox")
    asyncio.run(
        sandbox.batch_evaluate_results(
            input_files=cfg.input_files,
            answer_format="lean4-proof",
            **eval_config_dict,
        )
    )


def eval_lean4_statement(cfg):
    eval_config = LeanEvaluatorConfig(**cfg.eval_config)

    sandbox = get_sandbox(**eval_config.sandbox)
    eval_config_dict = asdict(eval_config)
    eval_config_dict.pop("sandbox")
    asyncio.run(
        sandbox.batch_evaluate_results(
            input_files=cfg.input_files,
            answer_format="lean4-statement",
            **eval_config_dict,
        )
    )
