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

import json

from nemo_skills.evaluation.evaluator.mcq import eval_mcq
from nemo_skills.evaluation.math_grader import extract_answer

BOTH = "Some reasoning with \\boxed{A} in the middle.\n\nThe final answer is B"
REGEX_ONLY = "Some reasoning.\n\nThe final answer is B"
BOXED_ONLY = "Some reasoning.\n\n\\boxed{A}"


def test_strict_extraction_uses_a_single_method():
    assert extract_answer(BOTH, extract_from_boxed=True) == "A"
    assert extract_answer(BOTH, extract_from_boxed=False) == "B"
    assert extract_answer(REGEX_ONLY, extract_from_boxed=True) is None
    assert extract_answer(BOXED_ONLY, extract_from_boxed=False) is None


def test_relaxed_extraction_prefers_boxed_by_default():
    assert extract_answer(BOTH, relaxed=True) == "A"
    assert extract_answer(REGEX_ONLY, relaxed=True) == "B"
    assert extract_answer(BOXED_ONLY, relaxed=True) == "A"


def test_relaxed_extraction_regex_first_is_opt_in():
    assert extract_answer(BOTH, relaxed=True, regex_first=True) == "B"
    assert extract_answer(REGEX_ONLY, relaxed=True, regex_first=True) == "B"
    assert extract_answer(BOXED_ONLY, relaxed=True, regex_first=True) == "A"
    # regex_first has no effect in strict mode
    assert extract_answer(BOTH, extract_from_boxed=True, regex_first=True) == "A"


def _mcq_predict(tmp_path, generation, eval_config=None, **sample_overrides):
    input_file = tmp_path / "output.jsonl"
    sample = {"generation": generation, "expected_answer": "A", **sample_overrides}
    input_file.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    eval_mcq({"input_file": str(input_file), **(eval_config or {})})
    return json.loads(input_file.read_text(encoding="utf-8"))["predicted_answer"]


def test_mcq_relaxed_order(tmp_path):
    generation = "Reasoning with \\boxed{A}.\n\nThe correct answer choice is: B"
    regex = r"(?i)The correct answer choice is\:*\**\s*\**([A-Za-z])"
    # default: boxed first
    assert _mcq_predict(tmp_path, generation, {"extract_regex": regex}) == "A"
    # opt-in via eval config
    assert _mcq_predict(tmp_path, generation, {"extract_regex": regex, "regex_first": True}) == "B"
    # opt-in via per-sample override
    assert _mcq_predict(tmp_path, generation, {"extract_regex": regex}, regex_first=True) == "B"
