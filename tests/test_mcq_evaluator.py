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

import pytest

from nemo_skills.evaluation.evaluator.mcq import eval_mcq


def _predict(tmp_path, generation, **sample_overrides):
    input_file = tmp_path / "output.jsonl"
    sample = {"generation": generation, "expected_answer": "D", **sample_overrides}
    input_file.write_text(json.dumps(sample) + "\n", encoding="utf-8")
    eval_mcq({"input_file": str(input_file)})
    return json.loads(input_file.read_text(encoding="utf-8"))["predicted_answer"]


@pytest.mark.parametrize(
    "generation, expected",
    [
        # explicit "Answer: X" line must win over a letter inside a boxed math expression (issue #1511)
        (
            "The adiabatic regime requires\n\n\\boxed{\\langle H\\rangle \\ll \\Delta E}\n\nso the correct statement is D.\n\n**Answer: D**",
            "D",
        ),
        # a boxed single letter is still used as is
        ("Reasoning...\n\n\\boxed{D}", "D"),
        ("Reasoning...\n\nAnswer: \\boxed{B}", "B"),
        # multi-character boxed content still falls back to the wrapped-letter heuristic when there is no explicit answer line
        ("Reasoning...\n\n\\boxed{(C)}", "C"),
        ("Reasoning...\n\n\\boxed{\\text{A}}", "A"),
        # explicit answer line with no boxed expression at all
        ("Reasoning...\n\nAnswer: B", "B"),
        # nothing parseable
        ("I am not sure.", None),
    ],
)
def test_extract_letter(tmp_path, generation, expected):
    assert _predict(tmp_path, generation) == expected
