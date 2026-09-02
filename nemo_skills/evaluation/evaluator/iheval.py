# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""IHEval evaluators — thin wrappers that score the jsonl in place via the external
``iheval`` package (https://github.com/bzantium/iheval), lazily imported like ``bfcl_eval``."""

import json
import logging

from nemo_skills.evaluation.evaluator.base import BaseEvaluatorConfig
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))

_INSTALL_HINT = (
    "The 'iheval' package is required to evaluate IHEval benchmarks. "
    "Install it with: pip install git+https://github.com/bzantium/iheval.git"
)


def _get_scorer(task: str):
    try:
        from iheval import SCORERS
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return SCORERS[task]


def _score_in_place(cfg, task: str):
    cfg = BaseEvaluatorConfig(**cfg)
    scorer = _get_scorer(task)
    jsonl_file = cfg.input_file

    with open(jsonl_file, "rt", encoding="utf-8") as fin:
        rows = [json.loads(line) for line in fin]

    for row in rows:
        scorer(row)

    with open(jsonl_file, "wt", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def eval_rule_following(cfg):
    _score_in_place(cfg, "rule_following")


def eval_verb_extract(cfg):
    _score_in_place(cfg, "verb_extract")


def eval_translation(cfg):
    _score_in_place(cfg, "translation")


def eval_lang_detect(cfg):
    _score_in_place(cfg, "lang_detect")


def eval_safety(cfg):
    _score_in_place(cfg, "safety")


def eval_slack_user(cfg):
    _score_in_place(cfg, "slack_user")


def eval_webpage(cfg):
    _score_in_place(cfg, "webpage")
