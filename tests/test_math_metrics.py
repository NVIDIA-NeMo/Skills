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

from nemo_skills.evaluation.metrics.math_metrics import MathMetrics


def _prediction(answer, is_correct, reward_score):
    return {
        "predicted_answer": answer,
        "judgement": f"Judgement: {'Yes' if is_correct else 'No'}",
        "reward_model_score": reward_score,
    }


@pytest.mark.parametrize(
    "verdicts,expected",
    [
        # The reward winner is "42" either way, so the order the judge disagreed in
        # must not change its score (#1257).
        ([True, False], 0.5),
        ([False, True], 0.5),
        ([True, True], 1.0),
        ([False, False], 0.0),
    ],
)
def test_reward_at_k_averages_verdicts_for_the_winning_answer(verdicts, expected):
    """Repeated occurrences of one answer are pooled instead of last-write-wins."""
    metrics = MathMetrics(compute_no_answer=False)
    predictions = [_prediction("42", verdict, reward_score=1.0) for verdict in verdicts]
    metrics._compute_reward_at_k(predictions)
    assert metrics.eval_dict["rm_majority@2"]["judge_correct"] == pytest.approx(expected)


def test_reward_at_k_picks_the_highest_cumulative_reward_answer():
    """Answer grouping does not disturb which answer wins on reward."""
    metrics = MathMetrics(compute_no_answer=False)
    predictions = [
        _prediction("42", is_correct=True, reward_score=0.6),
        _prediction("42", is_correct=True, reward_score=0.6),
        _prediction("43", is_correct=False, reward_score=1.0),
    ]
    metrics._compute_reward_at_k(predictions)
    # "42" has 1.2 cumulative reward against "43"'s 1.0, and both its verdicts agree.
    assert metrics.eval_dict["rm_majority@3"]["judge_correct"] == pytest.approx(1.0)
    # rm_best@k still follows the single highest-scoring generation, which is "43".
    assert metrics.eval_dict["rm_best@3"]["judge_correct"] == pytest.approx(0.0)
