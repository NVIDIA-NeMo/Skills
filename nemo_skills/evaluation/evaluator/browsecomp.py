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

# Parses the BrowseComp LLM-judge output (the `judgement` field) into a clean
# `judge_correct` boolean so that BrowseCompMetrics stays trivial.
# Grading logic follows the BrowseComp evaluation orchestrator (_parse_grading).

import json
import logging
import re

from nemo_skills.evaluation.evaluator.base import BaseEvaluatorConfig
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))

_CORRECT_RE = re.compile(r"correct:\s*(yes|no)\b", re.IGNORECASE)
_EXTRACTED_RE = re.compile(r"extracted_final_answer:\s*(.+?)(?:\n|$)")
# If the extracted answer still contains the template instruction, the grader echoed the prompt.
_TEMPLATE_ECHO = "The final exact answer extracted from the [response]"


def parse_grading(judgement):
    """Parse grader output. Returns (is_correct, extracted_answer, parsed_ok).

    Uses the LAST 'correct: yes/no' match to avoid picking up template echoes or
    reasoning inside <think> blocks produced by reasoning judge models.
    An unparseable verdict is treated as incorrect.
    """
    if not judgement:
        return False, None, False

    matches = list(_CORRECT_RE.finditer(judgement))
    if not matches:
        return False, None, False

    is_correct = matches[-1].group(1).lower() == "yes"

    ans_matches = list(_EXTRACTED_RE.finditer(judgement))
    extracted = ans_matches[-1].group(1).strip() if ans_matches else None

    if extracted and _TEMPLATE_ECHO in extracted:
        return False, None, False

    return is_correct, extracted, True


def eval_browsecomp(cfg):
    """Grade BrowseComp judge outputs in place, adding `judge_correct` / `extracted_answer`."""
    cfg = BaseEvaluatorConfig(**cfg)
    jsonl_file = cfg.input_file

    with open(jsonl_file, "rt", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]

    for sample in data:
        is_correct, extracted, parsed_ok = parse_grading(sample.get("judgement"))
        sample["judge_correct"] = bool(parsed_ok and is_correct)
        sample["extracted_answer"] = extracted

    with open(jsonl_file, "wt", encoding="utf-8") as fout:
        for sample in data:
            fout.write(json.dumps(sample) + "\n")

    LOG.info("BrowseComp grading completed for %s (%d samples)", jsonl_file, len(data))
