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

from typing import Any, Callable, Dict

from nemo_skills.evaluation.evaluator.bfcl import eval_bfcl
from nemo_skills.evaluation.evaluator.code import (
    eval_bigcodebench,
    eval_evalplus,
    eval_livebench_coding,
    eval_livecodebench,
    eval_livecodebench_pro,
)
from nemo_skills.evaluation.evaluator.ifbench import eval_ifbench
from nemo_skills.evaluation.evaluator.ifeval import eval_if
from nemo_skills.evaluation.evaluator.ioi import eval_ioi
from nemo_skills.evaluation.evaluator.math import (
    eval_lean4_proof,
    eval_lean4_statement,
    eval_math,
)
from nemo_skills.evaluation.evaluator.mcq import eval_mcq
from nemo_skills.evaluation.evaluator.mrcr import eval_mrcr
from nemo_skills.evaluation.evaluator.ruler import eval_ruler
from nemo_skills.evaluation.evaluator.scicode import eval_scicode


def dummy_eval(cfg):
    return


EVALUATOR_MAP = {
    "math": eval_math,
    "evalplus": eval_evalplus,
    "if": eval_if,
    "ifbench": eval_ifbench,
    "bfcl": eval_bfcl,
    "no-op": dummy_eval,
    "lean4-proof": eval_lean4_proof,
    "lean4-statement": eval_lean4_statement,
    "multichoice": eval_mcq,
    "ruler": eval_ruler,
    "livecodebench": eval_livecodebench,
    "livebench_coding": eval_livebench_coding,
    "livecodebench_pro": eval_livecodebench_pro,
    "scicode": eval_scicode,
    "mrcr": eval_mrcr,
    "ioi": eval_ioi,
    "bigcodebench": eval_bigcodebench,
}


def is_evaluator_registered(eval_type: str):
    return eval_type in EVALUATOR_MAP


def register_evaluator(eval_type: str, eval_fn: Callable[[Dict[str, Any]], None]):
    if is_evaluator_registered(eval_type):
        raise ValueError(f"Evaluator for {eval_type} already registered")

    EVALUATOR_MAP[eval_type] = eval_fn


def evaluate(cfg):
    if cfg.eval_type not in EVALUATOR_MAP:
        raise ValueError(
            f"Evaluator not found for type: {cfg.eval_type}.\nSupported types: {str(EVALUATOR_MAP.keys())}"
        )
    return EVALUATOR_MAP[cfg.eval_type](cfg)
