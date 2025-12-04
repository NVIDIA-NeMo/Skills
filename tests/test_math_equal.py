# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import pytest

from nemo_skills.evaluation.math_grader import math_equal


@pytest.mark.parametrize(
    "output_pair",
    [
        (5, 5),
        (5, 5.0),
        ("1/2", 0.5),
        ("\\frac{1}{2}", 0.5),
        ("918\\frac{1}{2}", 918.5),
        ("x^2+2x+1", "x^2 + 2*x + 1"),
        ("x^2+2x+1", "x^2 + 2*x - (-1)"),
        ("y = x^2+2x+1", "2x+ 1+x^2"),
        ("E", "\\mathrm{E}"),
        ("A", "\\textbf{A}"),
        ("f'", "f'"),
        ("185", "185\\"),
        ("185\\", "185\\"),
        (".185", "0.185"),
        ("016", "16"),  # Leading zeros handled by math_verify
        ("007", "7"),  # Multiple leading zeros handled by math_verify
        ("\\frac {1}{2}", 0.5),
        ("17\\text{ any text}", "17"),
        ("\$10", "10"),
        ("10%", "10"),
        ("10\\%", "10"),
        (5 / 2, "\\frac{5}{2}"),
        ("\\frac{1}{3}", "\\dfrac{1}{3}"),
        ("(r+5)(r+5)", "(r+5)^2"),
        ("\\frac{\\sqrt{3}}{3}", "\\frac{\\sqrt{3}}{3} \\approx 0.577"),
        (
            "\\begin{pmatrix}0&0&0\\\\0&1&0\\\\0&0&1\\end{pmatrix}",
            "\\begin{bmatrix} 0 & 0 & 0 \\\\ 0 & 1 & 0 \\\\ 0 & 0 & 1 \\end{bmatrix}",
        ),
    ],
    ids=str,
)
def test_correct_examples(output_pair):
    output = math_equal(output_pair[0], output_pair[1])
    assert output is True
    output = math_equal(output_pair[1], output_pair[0])
    assert output is True


@pytest.mark.parametrize(
    "output_pair",
    [
        (5, 5.001),
        (0, None),
        ("x^2+2x+1", "x^3+2x+1"),
        ("odd", "\\text{oddd}"),
        ("E", "\\mathrm{E}*2"),
        ("\\sqrt{67},-\\sqrt{85}", "\\sqrt{67}"),
    ],
    ids=str,
)
def test_incorrect_examples(output_pair):
    output = math_equal(output_pair[0], output_pair[1])
    assert output is False
    output = math_equal(output_pair[1], output_pair[0])
    assert output is False
