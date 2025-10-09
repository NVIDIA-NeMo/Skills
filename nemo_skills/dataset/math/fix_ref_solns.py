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

import re


def _post_fix(problem_id, soln_string):
    """Post fixing some answer strings"""
    if problem_id == "test/intermediate_algebra/78.json":
        soln_string = re.sub(r"\\(\d+)", r"\1", soln_string)

    if problem_id == "train/number_theory/7115.json":
        return "A"

    if problem_id == "train/number_theory/1012.json":
        return "E"

    if problem_id == "train/prealgebra/666.json":
        return "125"

    if problem_id == "train/intermediate_algebra/172.json":
        return "two lines"

    if problem_id == "train/prealgebra/1691.json":
        return "1.85"

    if problem_id == "train/geometry/6177.json":
        return "C"

    if problem_id == "train/number_theory/7117.json":
        return "A"

    if problem_id == "train/geometry/6202.json":
        return "D"

    if problem_id == "train/precalculus/268.json":
        return "A"

    return soln_string


def _post_fix_multi_answer(problem_id, results):
    """Fixing cases where there are multiple boxed entries."""

    if problem_id == "train/prealgebra/452.json":
        # Two ptions are mathematically equivalent
        return results[0]

    if problem_id == "train/algebra/1771.json":
        return ";".join(results)

    if problem_id == "train/algebra/152.json":
        return ";".join(results)

    if problem_id == "train/algebra/2156.json":
        # Same fraction but without antlr-4.11 can't verify if the two are equal
        return results[-1]

    if problem_id == "train/intermediate_algebra/1609.json":
        # Same fraction but without antlr-4.11 can't verify if the two are equal
        return results[-1]

    if problem_id == "train/precalculus/865.json":
        # The question has 2 answers, we just choose the last answer for now
        # TODO - Fix handling of OR questions
        return results[-1]

    if problem_id == "train/precalculus/982.json":
        # Question has many answers since it's a phase shift question.
        # Choosing the rightmost answer for now.
        # TODO - Fix handling of OR questions
        return results[-1]

    if problem_id == "train/precalculus/1149.json":
        # This question has two solutions, each being 30,150. We can pick any of the results entries
        return results[-1]

    if problem_id == "train/number_theory/837.json":
        # The two answers are for 24 hr clock vs 12 hr clock. Choosing the 24 version
        return results[0]

    if problem_id == "train/intermediate_algebra/396.json":
        # Or question, picking the rightmost answer
        return results[-1]

    if problem_id == "train/counting_and_probability/955.json":
        # The first boxed entry is an intermediate step
        return results[-1]

    # Test set fixes
    if problem_id == "test/prealgebra/1088.json":
        # Two solutions are mathematically equivalent
        return results[0]

    if problem_id == "test/algebra/1197.json":
        # The first entry is an intermediate result
        return results[-1]

    if problem_id == "test/geometry/66.json":
        # The two entries are same, choosing the first one expressed in frac
        return results[0]

    if problem_id == "test/geometry/1125.json":
        # Both are 0.25, choosing the first one
        return results[0]

    if problem_id == "test/prealgebra/1407.json":
        # There are intermediate values which are not answers
        return results[-1]

    if problem_id == "test/prealgebra/224.json":
        # Two answers are same, choosing rightmost
        return results[-1]

    if problem_id == "test/prealgebra/177.json":
        # The answer is 12 the last entry
        return results[-1]

    if problem_id == "test/number_theory/459.json":
        # Two answers are same, choosing rightmost
        return results[-1]

    if problem_id == "test/intermediate_algebra/702.json":
        # OR question. Choosing the rightmost answer
        return results[-1]

    if problem_id == "test/intermediate_algebra/25.json":
        # OR question. Choosing the rightmost answer
        return results[-1]

    if problem_id == "test/intermediate_algebra/747.json":
        # OR question. Choosing the rightmost answer
        return results[-1]

    return ",".join(results)


def _fix_solution(problem_id, ref_soln):
    if problem_id == "train/algebra/24014.json":
        return ref_soln.replace("$\\boxed 2$", "$\\boxed{2}$")

    if problem_id == "train/algebra/25040.json":
        return ref_soln.replace("\\boxed 9$", "\\boxed{9}$")

    if problem_id == "train/algebra/535.json":
        # Original soln
        # $\\log_24=\\boxed{2}$, so $\\log_2(4^2) = \\log_2((2^2)^2) = \\log_2 (2^4) = \\boxed{4}$
        return "$\\log_24=2$, so $\\log_2(4^2) = \\log_2((2^2)^2) = \\log_2 (2^4) = \\boxed{4}$"

    if problem_id == "train/geometry/892.json":
        return ref_soln.replace("\\boxed{144}", "144")

    if problem_id == "train/number_theory/7041.json":
        return ref_soln.replace("\\boxed{j-i \\equiv 0 \\pmod{6}}", "j-i \\equiv 0 \\pmod{6}")

    if problem_id == "train/intermediate_algebra/1266.json":
        return ref_soln.replace(
            "(x^2-\\boxed{\\phantom{09}})(x^2-\\boxed{\\phantom{25}})", "(x^2-\\phantom{09})(x^2-\\phantom{25})"
        )

    return ref_soln
