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

minif2f_deepseek_fewshot = [
    {
        "header": "import Mathlib\n\nopen Complex Filter Function Metric Finset\nopen scoped BigOperators Topology\n\n",
        "informal_prefix": "/-- Expand the following expression: $7(3y+2)$ Show that it is 21y+14.-/\n",
        "formal_statement": "theorem mathd_algebra_182 (y : ℂ) : 7 * (3 * y + 2) = 21 * y + 14 := by\n",
        "formal_proof": "/- We apply the distributive property to get\\begin{align*}\n  7(3y+2) &= 7\\cdot 3y+7\\cdot 2\\\\\n  &= 21y+14.\n  \\end{align*}\n  -/\nring",
    },
    {
        "header": "import Mathlib\n\nopen Complex Filter Function Metric Finset\nopen scoped BigOperators Topology\n\n",
        "informal_prefix": "/-- What is the units digit of $19^{19}+99^{99}$? Show that it is 8.-/\n",
        "formal_statement": "theorem mathd_numbertheory_202 : (19 ^ 19 + 99 ^ 99) % 10 = 8 := by\n",
        "formal_proof": "/- The units digit of a power of an integer is determined by the units digit of the integer; that is, the tens digit, hundreds digit, etc... of the integer have no effect on the units digit of the result. In this problem, the units digit of $19^{19}$ is the units digit of $9^{19}$. Note that $9^1=9$ ends in 9, $9^2=81$ ends in 1, $9^3=729$ ends in 9, and, in general, the units digit of odd powers of 9 is 9, whereas the units digit of even powers of 9 is 1. Since both exponents are odd, the sum of their units digits is $9+9=18$, the units digit of which is $8.$\n  -/\napply Eq.refl",
    },
    {
        "header": "import Mathlib\n\nopen Complex Filter Function Metric Finset\nopen scoped BigOperators Topology\n\n",
        "informal_prefix": "/-- At each basketball practice last week, Jenny made twice as many free throws as she made at the previous practice.  At her fifth practice she made 48 free throws.  How many free throws did she make at the first practice? Show that it is 3.-/\n",
        "formal_statement": "theorem mathd_algebra_455 (x : ℝ) (h₀ : 2 * (2 * (2 * (2 * x))) = 48) : x = 3 := by\n",
        "formal_proof": "/- At Jenny's fourth practice she made $\\frac{1}{2}(48)=24$ free throws. At her third practice she made 12, at her second practice she made 6, and at her first practice she made $3$.\n  -/\nlinarith",
    },
    {
        "header": "import Mathlib\n\nopen Complex Filter Function Metric Finset\nopen scoped BigOperators Topology\n\n",
        "informal_prefix": "/-- A group of $N$ students, where $N < 50$, is on a field trip. If their teacher puts them in groups of 8, the last group has 5 students. If their teacher instead puts them in groups of 6, the last group has 3 students. What is the sum of all possible values of $N$? Show that it is 66.-/\n",
        "formal_statement": "theorem mathd_numbertheory_149 :\n  (∑ k in Finset.filter (fun x => x % 8 = 5 ∧ x % 6 = 3) (Finset.range 50), k) = 66 := by\n",
        "formal_proof": "/- We are given that $N\\equiv 5\\pmod{8}$ and $N\\equiv 3\\pmod{6}$.  We begin checking numbers which are 5 more than a multiple of 8, and we find that 5 and 13 are not 3 more than a multiple of 6, but 21 is 3 more than a multiple of 6. Thus 21 is one possible value of $N$. By the Chinese Remainder Theorem, the integers $x$ satisfying $x\\equiv 5\\pmod{8}$ and $x\\equiv 3\\pmod{6}$ are those of the form $x=21+\\text{lcm}(6,8)k = 21 + 24 k$, where $k$ is an integer. Thus the 2 solutions less than $50$ are 21 and $21+24(1) = 45$, and their sum is $21+45=66$.\n  -/\napply Eq.refl",
    },
    {
        "header": "import Mathlib\n\nopen Complex Filter Function Metric Finset\nopen scoped BigOperators Topology\n\n",
        "informal_prefix": "/-- Evaluate: $\\left( \\frac{1}{2} + \\frac{1}{3} \\right) \\left( \\frac{1}{2} - \\frac{1}{3} \\right)$ Show that it is \\frac{5}{36}.-/\n",
        "formal_statement": "theorem mathd_algebra_462 : ((1 : ℚ) / 2 + 1 / 3) * (1 / 2 - 1 / 3) = 5 / 36 := by\n",
        "formal_proof": "/- For any $x$ and $y$, $(x+y)(x-y)=x^2-y^2+xy-xy=x^2-y^2$, so \\begin{align*}\n  \\left( \\frac{1}{2} + \\frac{1}{3} \\right) \\left( \\frac{1}{2} - \\frac{1}{3} \\right)&=\\left(\\frac12\\right)^2-\\left(\\frac13\\right)^2\\\\\n  &=\\frac14-\\frac19\\\\\n  &=\\frac{9}{36}-\\frac{4}{36}\\\\\n  &=\\frac{5}{36}\n  \\end{align*}\n  -/\nsimp_all only [one_div]\nnorm_num",
    },
]

math_to_lean4_fewshot = [
    {
        "header": "import Mathlib\n\nopen Complex Filter Function Metric Finset\nopen scoped BigOperators Topology\n\n",
        "problem": "What is the following value when expressed as a common fraction: $$\\frac{1}{2^{1}}+\\frac{1}{2^{2}}+\\frac{1}{2^{3}}+\\cdots + \\frac{1}{2^{8}}+\\frac{1}{2^{9}}+\\frac{1}{2^{10}}?$$",
        "predicted_answer": "\\frac{1023}{1024}",
        "formal_statement": "theorem user_theorem : (\u2211 k in Finset.range 10, (1 / (2 ^ (k + 1)))) = 1023 / 1024 := by\n",
        "id": "test/algebra/2130.json",
        "formal_proof": "sorry",
    },
    {
        "header": "import Mathlib\n\nopen Complex Filter Function Metric Finset\nopen scoped BigOperators Topology\n\n",
        "problem": "Evaluate $24-(2x-y)$ if $x=4$ and $y=3$.",
        "predicted_answer": "19",
        "formal_statement": "theorem user_theorem : 24 - (2 * 4 - 3) = 19 := by\n",
        "id": "test/algebra/1264.json",
        "formal_proof": "sorry",
    },
    {
        "header": "import Mathlib\n\nopen Complex Filter Function Metric Finset\nopen scoped BigOperators Topology\n\n",
        "problem": "If $x+y=12$ and $x-y=8$, what is the value of $2x-xy$?",
        "predicted_answer": "0",
        "formal_statement": "theorem user_theorem (x y : \u211d) (h\u2080 : x + y = 12) (h\u2081 : x - y = 8) : 2 * x - x * y = 0 := by\n",
        "id": "test/algebra/1272.json",
        "formal_proof": "sorry",
    },
    {
        "header": "import Mathlib\n\nopen Complex Filter Function Metric Finset\nopen scoped BigOperators Topology\n\n",
        "problem": "A parabola with equation $y=x^2+bx+c$ passes through the points $(2,3)$ and $(4,3)$. What is $c$?",
        "predicted_answer": "11",
        "formal_statement": "theorem user_theorem (b c : \u211d) (h\u2081 : 3 = 2 ^ 2 + 2 * b + c) (h\u2082 : 3 = 4 ^ 2 + 4 * b + c) : c = 11 := by\n",
        "id": "test/algebra/636.json",
        "formal_proof": "sorry",
    },
    {
        "header": "import Mathlib\n\nopen Complex Filter Function Metric Finset\nopen scoped BigOperators Topology\n\n",
        "problem": "Two standard six-faced dice are rolled. Jean wins if the product of the two numbers rolled is odd or a multiple of three, otherwise Allen wins. What is the probability that Jean wins? Express your answer as a common fraction.",
        "predicted_answer": "\\frac{2}{3}",
        "formal_statement": "theorem user_theorem : ((Finset.filter (fun x => (x.1 * x.2) % 2 = 1 ∨ (x.1 * x.2) % 3 = 0) (Finset.product (Finset.Icc 1 6) (Finset.Icc 1 6))).card : ℚ) / (Finset.product (Finset.Icc 1 6) (Finset.Icc 1 6)).card = (2 : ℚ) / 3 := by\n",
        "id": "test_counting_and_probability/551.json",
        "formal_proof": "sorry",
    },
]

lean4_false_fewshots = [
    {
        "header": "import Mathlib\n\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen Topology Filter Real Complex TopologicalSpace Finset Function Metric Nat Rat\nopen scoped BigOperators Matrix\n\n",
        "informal_prefix": "/-- Prove that a real number cannot be both greater than and less than zero simultaneously. -/\n",
        "formal_statement": "theorem user_theorem : \n  ∀ (x : \u211d), x > 0 \u2227 x < 0 \u2192 False := by\n",
        "formal_proof": "/- Assume x > 0 and x < 0. The lt_asymm theorem states that for any real numbers a and b, both a < b and b < a cannot hold simultaneously. Applying this to x > 0 and x < 0 yields a contradiction. -/\nintro x h\nhave h1 : x > 0 := h.1\nhave h2 : x < 0 := h.2\nexact lt_asymm h1 h2",
    },
    {
        "header": "import Mathlib\n\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen Topology Filter Real Complex TopologicalSpace Finset Function Metric Nat Rat\nopen scoped BigOperators Matrix\n\n",
        "informal_prefix": "/-- Prove that a real number cannot be both less than and greater than another real number. -/\n",
        "formal_statement": "theorem user_theorem_4 : \n  ∀ (x y : \u211d), x < y \u2227 y < x \u2192 False := by\n",
        "formal_proof": "/- Assume x < y and y < x. The lt_asymm theorem ensures that this combination is contradictory. -/\nintros x y h\nhave h1 : x < y := h.1\nhave h2 : y < x := h.2\nexact lt_asymm h1 h2",
    },
    {
        "header": "import Mathlib\n\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen Topology Filter Real Complex TopologicalSpace Finset Function Metric Nat Rat\nopen scoped BigOperators Matrix\n\n",
        "informal_prefix": "/-- Prove that an integer cannot have a remainder of both 0 and 1 modulo 2. -/\n",
        "formal_statement": "theorem user_theorem_5 : \n  ∀ (x : \u2124), x % 2 = 0 \u2227 x % 2 = 1 \u2192 False := by\n",
        "formal_proof": "/- Assume x % 2 = 0 and x % 2 = 1. Substituting the first equality into the second leads to 0 = 1, which is a contradiction. -/\nintro x h\nhave h1 : x % 2 = 0 := h.1\nhave h2 : x % 2 = 1 := h.2\nrw [h1] at h2\nexact Int.zero_ne_one h2",
    },
    {
        "header": "import Mathlib\n\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen Topology Filter Real Complex TopologicalSpace Finset Function Metric Nat Rat\nopen scoped BigOperators Matrix\n\n",
        "informal_prefix": "/-- Prove that a real number cannot be both strictly greater than and less than or equal to another real number. -/\n",
        "formal_statement": "theorem user_theorem_6 : \n  ∀ (x y : \u211d), x > y \u2227 x ≤ y \u2192 False := by\n",
        "formal_proof": "/- Assume x > y and x ≤ y. The not_le_of_gt theorem ensures that this combination leads to a contradiction. -/\nintros x y h\nhave h1 : x > y := h.1\nhave h2 : x ≤ y := h.2\nexact not_le_of_gt h1 h2",
    },
    {
        "header": "import Mathlib\n\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen Topology Filter Real Complex TopologicalSpace Finset Function Metric Nat Rat\nopen scoped BigOperators Matrix\n\n",
        "informal_prefix": "/-- Prove that a real number cannot be both strictly positive and less than or equal to zero. -/\n",
        "formal_statement": "theorem user_theorem_8 : \n  ∀ (x : \u211d), x > 0 \u2227 x ≤ 0 \u2192 False := by\n",
        "formal_proof": "/- Assume x > 0 and x ≤ 0. The not_le_of_gt theorem ensures that this leads to a contradiction. -/\nintro x h\nhave h1 : x > 0 := h.1\nhave h2 : x ≤ 0 := h.2\nexact not_le_of_gt h1 h2",
    },
]


nat_nat_judgment_fewshots = [
    {
        "original_statement": "Solve for x: 4x - 8 = 0.",
        "backtranslation": "Show that x = 2 is the unique solution of 4x - 8 = 0.",
        "reasoning": "The first statement simply asks to solve the equation, while the backtranslation requires demonstrating that x = 2 is the only solution. Both tasks lead to the same result.",
        "judgment": "valid",
    },
    {
        "original_statement": "Find the roots of the equation x^2 - 9 = 0.",
        "backtranslation": "Prove that x = 3 and x = -3 are the only solutions of x^2 - 9 = 0.",
        "reasoning": "Although the second statement uses a 'prove that' phrasing, it ultimately requires identifying the same two roots as the first statement.",
        "judgment": "valid",
    },
    {
        "original_statement": "Determine the limit of (1/x) as x approaches infinity.",
        "backtranslation": "Show that the series 1/x diverges as x approaches infinity.",
        "reasoning": "The first statement asks for the limit of a function, which is 0, while the backtranslation mistakenly addresses a series and claims divergence. This changes the task entirely.",
        "judgment": "invalid",
    },
    {
        "original_statement": "Find x^2 + 2x + 1 = 0.",
        "backtranslation": "Show that x = -1 is the only solution of x^2 + 2x + 1 = 0.",
        "reasoning": "Both statements are focused on solving the quadratic equation. Despite the backtranslation framing it as a proof that x = -1 is the unique solution, the underlying problem remains the same.",
        "judgment": "valid",
    },
    {
        "original_statement": "Determine whether f(x) = x^2 is differentiable at x = 0.",
        "backtranslation": "Find the derivative of f(x) = x^2 at x = 0.",
        "reasoning": "The first statement asks for differentiability, which requires checking the existence of the derivative, whereas the second statement assumes differentiability and directly asks for the derivative. The tasks are different.",
        "judgment": "invalid",
    },
]


examples_map = {
    "minif2f_deepseek_fewshot": minif2f_deepseek_fewshot,
    "math_to_lean4_fewshot": math_to_lean4_fewshot,
    "lean4_false_fewshots": lean4_false_fewshots,
    "nat_nat_judgment_fewshots": nat_nat_judgment_fewshots,
}
