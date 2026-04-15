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

from nemo_skills.dataset.hle import HLE_ANSWER_EXTRACT_REGEX
from nemo_skills.evaluation.math_grader import extract_answer, math_equal


def extract_hle_answer(generation: str) -> str | None:
    return extract_answer(
        generation,
        extract_from_boxed=False,
        extract_regex=HLE_ANSWER_EXTRACT_REGEX,
        relaxed=True,
    )


def test_hle_extracts_plain_answer_line():
    generation = """Explanation: Arrhenius's theorem rules out this view.
Answer: D
Confidence: 100%"""

    assert extract_hle_answer(generation) == "D"


def test_hle_extracts_markdown_answer_line():
    generation = """
Based on my historical analysis, I can now provide the answer.

**The Answer is C. 1223, Rigord**

Here's the breakdown...

Answer: **C. 1223, Rigord**

Confidence: **90%**
"""

    extracted = extract_hle_answer(generation)

    assert extracted == "**C. 1223, Rigord**"
    assert math_equal("C", extracted)


def test_hle_falls_back_to_boxed_answer():
    generation = r"""Explanation: Using Chebotarev, the density is \boxed{\frac{2}{7}}."""

    assert extract_hle_answer(generation) == r"\frac{2}{7}"


def test_hle_extracts_exact_match_answer():
    generation = """Explanation: Putting the letters together gives the final string.
Answer: yeyo
Confidence: 100%"""

    extracted = extract_hle_answer(generation)

    assert extracted == "yeyo"
    assert math_equal("yeyo", extracted)
