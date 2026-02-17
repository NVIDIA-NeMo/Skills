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
import inspect
from typing import Any, Callable, Dict

from nemo_skills.dataset.utils import locate
from nemo_skills.evaluation.evaluator.audio import AudioEvaluator
from nemo_skills.evaluation.evaluator.base import BaseEvaluator
from nemo_skills.evaluation.evaluator.bfcl import eval_bfcl
from nemo_skills.evaluation.evaluator.bird import BirdEvaluator
from nemo_skills.evaluation.evaluator.code import (
    CodeExecEvaluator,
    eval_bigcodebench,
    eval_evalplus,
    eval_human_eval_infilling,
    eval_livebench_coding,
    eval_livecodebench_pro,
)
from nemo_skills.evaluation.evaluator.compute_eval import ComputeEvalEvaluator
from nemo_skills.evaluation.evaluator.icpc import ICPCEvaluator
from nemo_skills.evaluation.evaluator.ifbench import eval_ifbench
from nemo_skills.evaluation.evaluator.ifeval import eval_if
from nemo_skills.evaluation.evaluator.ioi import IOIEvaluator
from nemo_skills.evaluation.evaluator.livecodebench import eval_livecodebench
from nemo_skills.evaluation.evaluator.math import (
    Lean4ProofEvaluator,
    MathEvaluator,
)
from nemo_skills.evaluation.evaluator.mcq import eval_mcq
from nemo_skills.evaluation.evaluator.mmau_pro import eval_mmau_pro
from nemo_skills.evaluation.evaluator.mrcr import eval_mrcr
from nemo_skills.evaluation.evaluator.ruler import eval_ruler, eval_ruler2
from nemo_skills.evaluation.evaluator.scicode import eval_scicode

EVALUATOR_MAP = {
    # Function-based evaluators (batch-only)
    "evalplus": eval_evalplus,
    "if": eval_if,
    "ifbench": eval_ifbench,
    "bfcl": eval_bfcl,
    "multichoice": eval_mcq,
    "ruler": eval_ruler,
    "ruler2": eval_ruler2,
    "livecodebench": eval_livecodebench,
    "livebench_coding": eval_livebench_coding,
    "livecodebench_pro": eval_livecodebench_pro,
    "scicode": eval_scicode,
    "mrcr": eval_mrcr,
    "bigcodebench": eval_bigcodebench,
    "human_eval_infilling": eval_human_eval_infilling,
    "mmau-pro": eval_mmau_pro,
}

# Evaluator class mapping, other evaluators can be added here as they're converted to classes
EVALUATOR_CLASS_MAP = {
    "math": MathEvaluator,
    "lean4-proof": Lean4ProofEvaluator,
    "code_exec": CodeExecEvaluator,
    "ioi": IOIEvaluator,
    "icpc": ICPCEvaluator,
    "audio": AudioEvaluator,
    "bird": BirdEvaluator,
    "compute-eval": ComputeEvalEvaluator,
}

# Validation: Ensure no overlap between class and function maps
_class_types = set(EVALUATOR_CLASS_MAP.keys())
_function_types = set(EVALUATOR_MAP.keys())
_overlap = _class_types.intersection(_function_types)
if _overlap:
    raise ValueError(
        f"Evaluator types cannot be in both EVALUATOR_CLASS_MAP and EVALUATOR_MAP: {_overlap}. "
        f"Each eval_type must be in exactly one map."
    )


def _resolve_eval_type(eval_type: str):
    """Resolve eval_type to either a class or function.

    Supports two formats:
        - Built-in string key: looks up in EVALUATOR_CLASS_MAP / EVALUATOR_MAP
        - Path format with `::`: `module.path::name` or `/path/to/file.py::name`
          Dynamically imports the module and returns the attribute.

    Returns (obj, is_class) where is_class is True if obj is a BaseEvaluator subclass.
    Returns (None, False) if eval_type is a plain string not found in either map.
    """
    if "::" in eval_type:
        obj = locate(eval_type)
        is_class = inspect.isclass(obj) and issubclass(obj, BaseEvaluator)
        return obj, is_class

    if eval_type in EVALUATOR_CLASS_MAP:
        return EVALUATOR_CLASS_MAP[eval_type], True
    if eval_type in EVALUATOR_MAP:
        return EVALUATOR_MAP[eval_type], False
    return None, False


def is_evaluator_registered(eval_type: str):
    """Check if evaluator is registered in either class or function map."""
    return eval_type in EVALUATOR_CLASS_MAP or eval_type in EVALUATOR_MAP


def register_evaluator(eval_type: str, eval_fn: Callable[[Dict[str, Any]], None], ignore_if_registered: bool = False):
    if is_evaluator_registered(eval_type):
        if ignore_if_registered:
            return
        raise ValueError(f"Evaluator for {eval_type} already registered")

    EVALUATOR_MAP[eval_type] = eval_fn


def get_evaluator_class(eval_type: str, config: Dict[str, Any]) -> BaseEvaluator:
    """Get evaluator instance by type."""
    obj, is_class = _resolve_eval_type(eval_type)
    if obj is None or not is_class:
        raise ValueError(
            f"Evaluator class not found for type: {eval_type}.\n"
            f"Available types with class support: {list(EVALUATOR_CLASS_MAP.keys())}\n"
            f"All supported types: {list(EVALUATOR_MAP.keys())}\n"
            f"Or use path format: module.path::ClassName or /path/to/file.py::ClassName"
        )
    return obj(config)


def supports_single_eval(eval_type: str, config: Dict[str, Any]) -> bool:
    """Check if evaluator supports single data point evaluation during generation."""
    obj, is_class = _resolve_eval_type(eval_type)
    if not is_class:
        return False  # Only class-based evaluators support single eval

    evaluator = obj(config)
    return evaluator.supports_single_eval()


def evaluate(eval_type, eval_config):
    """Main evaluation function that handles both class-based and function-based evaluators.

    eval_type can be a built-in string key or a path in the format:
        - module.path::name (for importable modules)
        - /path/to/file.py::name (for file-based imports)
    """
    obj, is_class = _resolve_eval_type(eval_type)

    if obj is None:
        all_types = list(EVALUATOR_CLASS_MAP.keys()) + list(EVALUATOR_MAP.keys())
        raise ValueError(
            f"Evaluator not found for type: {eval_type}.\n"
            f"Supported types: {sorted(all_types)}\n"
            f"Or use path format: module.path::name or /path/to/file.py::name"
        )

    if is_class:
        evaluator = obj(eval_config)
        return asyncio.run(evaluator.eval_full())

    return obj(eval_config)
