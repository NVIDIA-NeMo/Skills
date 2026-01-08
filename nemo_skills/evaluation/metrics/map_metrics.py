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
# See the License for the specific lang

import functools
import importlib
from pathlib import Path

from nemo_skills.dataset.utils import import_from_path
from nemo_skills.evaluation.metrics.aalcr_metrics import AALCRMetrics
from nemo_skills.evaluation.metrics.answer_judgement_metrics import AnswerJudgementMetrics
from nemo_skills.evaluation.metrics.arena_metrics import ArenaMetrics
from nemo_skills.evaluation.metrics.audio_metrics import AudioMetrics
from nemo_skills.evaluation.metrics.bfcl_metrics import BFCLMetrics
from nemo_skills.evaluation.metrics.bird_metrics import BirdMetrics
from nemo_skills.evaluation.metrics.code_metrics import (
    BigCodeBenchMetrics,
    EvalPlusMetrics,
    HumanEvalInfillingMetrics,
    LiveCodeBenchMetrics,
    SciCodeMetrics,
    SweBenchMetrics,
)
from nemo_skills.evaluation.metrics.icpc_metrics import ICPCMetrics
from nemo_skills.evaluation.metrics.if_metrics import IFMetrics
from nemo_skills.evaluation.metrics.ioi_metrics import IOIMetrics
from nemo_skills.evaluation.metrics.lean4_metrics import Lean4Metrics
from nemo_skills.evaluation.metrics.math_metrics import MathMetrics
from nemo_skills.evaluation.metrics.mmau_pro_metrics import MMAUProMetrics
from nemo_skills.evaluation.metrics.mrcr_metrics import MRCRMetrics
from nemo_skills.evaluation.metrics.ruler_metrics import RulerMetrics
from nemo_skills.evaluation.metrics.simpleqa_metrics import SimpleQAMetrics
from nemo_skills.evaluation.metrics.translation_metrics import TranslationMetrics

METRICS_MAP = {
    "math": MathMetrics,
    "hle": functools.partial(MathMetrics, compute_no_answer=False, answer_key="generation"),
    "simpleqa": SimpleQAMetrics,
    "lean4-proof": Lean4Metrics,
    "lean4-statement": Lean4Metrics,
    "answer-judgement": AnswerJudgementMetrics,
    "arena": ArenaMetrics,
    "audio": AudioMetrics,
    "bfcl": BFCLMetrics,
    "bird": BirdMetrics,
    "evalplus": EvalPlusMetrics,
    "if": IFMetrics,
    "ioi": IOIMetrics,
    "icpc": ICPCMetrics,
    "multichoice": MathMetrics,
    "ruler": RulerMetrics,
    "livecodebench": LiveCodeBenchMetrics,
    "livecodebench_pro": LiveCodeBenchMetrics,
    "swe-bench": SweBenchMetrics,
    "scicode": SciCodeMetrics,
    "bigcodebench": BigCodeBenchMetrics,
    "mrcr": MRCRMetrics,
    "aalcr": AALCRMetrics,
    "livebench_coding": LiveCodeBenchMetrics,
    "translation": TranslationMetrics,
    "human_eval_infilling": HumanEvalInfillingMetrics,
    "mmau_pro_closed_form": MMAUProMetrics,
    "mmau_pro_open_ended": MMAUProMetrics,
    "mmau_pro_instruction_following": MMAUProMetrics,
}


def get_metrics(metric_type: str, **kwargs):
    """Get metrics class.

    Class path formats:
        - Module format: `path.to.module::ClassName`
        - Path format: `/path/to/module/file.py::ClassName`

    Arguments:
        metric_type: Either a string from METRICS_MAP, or a path to class (class path format above).
        **kwargs: Additional kwargs to pass to the metrics class constructor.
    """
    metrics_cls = None

    if metric_type in METRICS_MAP:
        metrics_cls = METRICS_MAP[metric_type]
    elif "::" in metric_type:
        module_str, class_str = metric_type.split("::", 1)
        if Path(module_str).is_file():
            module = import_from_path(module_str)
        else:
            module = importlib.import_module(module_str)

        metrics_cls = getattr(module, class_str)

    if metrics_cls is None:
        raise ValueError(
            f"Metric {metric_type} not found.\nSupported types: {str(METRICS_MAP.keys())} or use explicit class path format."
        )

    return metrics_cls(**kwargs)
