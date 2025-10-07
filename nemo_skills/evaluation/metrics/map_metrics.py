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

from nemo_skills.evaluation.metrics.aalcr_metrics import AALCRMetrics
from nemo_skills.evaluation.metrics.answer_judgement_metrics import AnswerJudgementMetrics
from nemo_skills.evaluation.metrics.arena_metrics import ArenaMetrics
from nemo_skills.evaluation.metrics.bfcl_metrics import BFCLMetrics
from nemo_skills.evaluation.metrics.code_metrics import (
    BigCodeBenchMetrics,
    EvalPlusMetrics,
    HumanEvalInfillingMetrics,
    LiveCodeBenchMetrics,
    OJBenchMetrics,
    SciCodeMetrics,
    SweBenchMetrics,
)
from nemo_skills.evaluation.metrics.if_metrics import IFMetrics
from nemo_skills.evaluation.metrics.ioi_metrics import IOIMetrics
from nemo_skills.evaluation.metrics.lean4_metrics import Lean4Metrics
from nemo_skills.evaluation.metrics.math_metrics import MathMetrics
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
    "bfcl": BFCLMetrics,
    "evalplus": EvalPlusMetrics,
    "if": IFMetrics,
    "ioi": IOIMetrics,
    "multichoice": MathMetrics,
    "ruler": RulerMetrics,
    "livecodebench": LiveCodeBenchMetrics,
    "swe-bench": SweBenchMetrics,
    "scicode": SciCodeMetrics,
    "bigcodebench": BigCodeBenchMetrics,
    "mrcr": MRCRMetrics,
    "aalcr": AALCRMetrics,
    "livebench_coding": LiveCodeBenchMetrics,
    "ojbench": OJBenchMetrics,
    "translation": TranslationMetrics,
    "human_eval_infilling": HumanEvalInfillingMetrics,
}


def get_metrics(metric_type: str):
    if metric_type not in METRICS_MAP:
        raise ValueError(f"Metric {metric_type} not found.\nSupported types: {str(METRICS_MAP.keys())}")
    return METRICS_MAP[metric_type]()
