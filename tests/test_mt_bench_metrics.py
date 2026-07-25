# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import math

import pytest

from nemo_skills.evaluation.metrics.mt_bench_metrics import MTBenchMetrics


def _pred(turn1, turn2, *, category="writing", target_language=None, answer_1=None, answer_2=None):
    pred = {
        "category": category,
        "judgement_turn1": f"[[{turn1}]]" if turn1 is not None else "",
        "judgement_turn2": f"[[{turn2}]]" if turn2 is not None else "",
    }
    if target_language is not None:
        pred["target_language"] = target_language
    if answer_1 is not None:
        pred["answer_1"] = answer_1
    if answer_2 is not None:
        pred["answer_2"] = answer_2
    return pred


# ---- Plain MT-Bench (no target_language) ----------------------------------


def test_mt_bench_single_sample_per_row():
    m = MTBenchMetrics()
    m.update([_pred(8.0, 7.0, category="writing")])
    m.update([_pred(6.0, 5.0, category="math")])

    metrics = m.get_metrics()
    assert list(metrics.keys()) == ["pass@1"]
    overall = metrics["pass@1"]
    assert overall["num_entries"] == 2
    assert overall["average_score"] == (8.0 + 7.0 + 6.0 + 5.0) / 4
    assert overall["turn1_average"] == 7.0
    assert overall["turn2_average"] == 6.0
    assert overall["category_writing"] == 7.5
    assert overall["category_math"] == 5.5
    # No target_language anywhere — raw_* should not be emitted.
    assert "raw_average_score" not in overall
    assert m.evaluations_to_print() == ["pass@1"]


def test_mt_bench_task_n_flattens_samples():
    m = MTBenchMetrics()
    n = 3
    m.update([_pred(s, s, category="writing") for s in (4.0, 6.0, 8.0)])

    metrics = m.get_metrics()
    assert m.total == 1
    assert m.max_k == n
    assert len(m.entries) == n
    # Single-key output: pass@1 already aggregates over all N samples.
    assert list(metrics.keys()) == ["pass@1"]
    overall = metrics["pass@1"]
    assert overall["turn1_average"] == 6.0
    assert overall["turn2_average"] == 6.0
    assert overall["average_score"] == 6.0
    assert m.evaluations_to_print() == ["pass@1"]


def test_mt_bench_multi_judge_averaging():
    m = MTBenchMetrics()
    pred = {
        "category": "writing",
        "all_judgements_turn1": ["[[8]]", "[[6]]", "[[10]]"],
        "all_judgements_turn2": ["[[5]]", "[[7]]"],
    }
    m.update([pred])
    overall = m.get_metrics()["pass@1"]
    assert overall["turn1_average"] == 8.0
    assert overall["turn2_average"] == 6.0


def test_mt_bench_invalid_ratings_dropped():
    m = MTBenchMetrics()
    pred = {
        "category": "writing",
        "judgement_turn1": "no rating here",
        "judgement_turn2": "[[15]]",  # out of 1-10 range
    }
    m.update([pred])
    overall = m.get_metrics()["pass@1"]
    assert "average_score" not in overall
    assert overall["num_entries"] == 1


# ---- target_language penalty (covers former KomtBench behavior) ----------

KO_ANSWER = "안녕하세요 반갑습니다 정말 좋은 하루 보내세요"
EN_ANSWER = "Hello, glad to meet you. Have a wonderful day."


def test_mt_bench_target_language_match_no_penalty():
    m = MTBenchMetrics()
    m.update([_pred(8.0, 6.0, target_language="ko", answer_1=KO_ANSWER, answer_2=KO_ANSWER)])
    overall = m.get_metrics()["pass@1"]
    assert overall["turn1_average"] == 8.0
    assert overall["turn2_average"] == 6.0
    # Penalty path was active so raw_* should be emitted (and equal primary).
    assert overall["raw_turn1_average"] == 8.0
    assert overall["raw_turn2_average"] == 6.0


def test_mt_bench_target_language_mismatch_applies_sqrt_penalty():
    m = MTBenchMetrics()
    m.update([_pred(9.0, 4.0, target_language="ko", answer_1=EN_ANSWER, answer_2=EN_ANSWER)])
    overall = m.get_metrics()["pass@1"]
    assert math.isclose(overall["turn1_average"], math.sqrt(9.0))
    assert math.isclose(overall["turn2_average"], math.sqrt(4.0))
    assert overall["raw_turn1_average"] == 9.0
    assert overall["raw_turn2_average"] == 4.0


def test_mt_bench_no_target_language_means_no_penalty():
    """Plain MT-Bench: target_language unset → penalty never applies even for English."""
    m = MTBenchMetrics()
    m.update([_pred(9.0, 4.0, answer_1=EN_ANSWER, answer_2=EN_ANSWER)])
    overall = m.get_metrics()["pass@1"]
    assert overall["turn1_average"] == 9.0
    assert overall["turn2_average"] == 4.0
    assert "raw_turn1_average" not in overall


def test_mt_bench_per_sample_penalty_under_task_n():
    """Penalty is applied per (row, sample), so mixed-language samples mix penalized and raw."""
    m = MTBenchMetrics()
    m.update(
        [
            _pred(9.0, 9.0, target_language="ko", answer_1=KO_ANSWER, answer_2=KO_ANSWER),
            _pred(9.0, 9.0, target_language="ko", answer_1=EN_ANSWER, answer_2=EN_ANSWER),
        ]
    )
    overall = m.get_metrics()["pass@1"]
    expected = (9.0 + math.sqrt(9.0)) / 2
    assert math.isclose(overall["turn1_average"], expected)
    assert math.isclose(overall["turn2_average"], expected)
    assert overall["raw_turn1_average"] == 9.0
    assert overall["raw_turn2_average"] == 9.0


def test_mt_bench_invalid_target_language_raises():
    m = MTBenchMetrics()
    with pytest.raises(ValueError, match="ISO 639-1"):
        m.update([_pred(8.0, 8.0, target_language="xx", answer_1=KO_ANSWER, answer_2=KO_ANSWER)])


def test_mt_bench_missing_required_fields_fail_fast():
    m = MTBenchMetrics()
    # Required single-judge fields use direct indexing, so missing keys raise.
    with pytest.raises(KeyError):
        m.update([{"category": "writing"}])
