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

import pytest

from recipes.sherloc.evaluation.sherloc import (
    calculate_line_overlap,
    evaluate_chunk_containment_metrics,
    evaluate_file_level_accuracy,
)


def _loc(file_path, start_line, end_line):
    """Build a single predicted or gold location."""
    return {"file_path": file_path, "start_line": start_line, "end_line": end_line}


@pytest.mark.parametrize(
    "gt_start,gt_end,pred_start,pred_end,expected",
    [
        (10, 20, 10, 20, 1.0),
        (10, 20, 15, 25, 6 / 16),
        (10, 20, 21, 30, 0.0),
        (10, 20, 20, 30, 1 / 21),
        (10, 10, 10, 10, 1.0),
        (10, 20, 1, 100, 11 / 100),
    ],
)
def test_calculate_line_overlap(gt_start, gt_end, pred_start, pred_end, expected):
    assert calculate_line_overlap(gt_start, gt_end, pred_start, pred_end) == pytest.approx(expected)


@pytest.mark.parametrize(
    "gt_start,gt_end,pred_start,pred_end",
    [(10, 20, 15, 25), (10, 20, 21, 30), (10, 20, 1, 100)],
)
def test_calculate_line_overlap_is_symmetric(gt_start, gt_end, pred_start, pred_end):
    assert calculate_line_overlap(gt_start, gt_end, pred_start, pred_end) == calculate_line_overlap(
        pred_start, pred_end, gt_start, gt_end
    )


def test_file_level_both_empty_scores_one():
    scores = evaluate_file_level_accuracy([], [])
    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["f1"] == 1.0


def test_file_level_one_sided_scores_zero():
    assert evaluate_file_level_accuracy([], [_loc("a.py", 1, 2)])["recall"] == 0.0
    assert evaluate_file_level_accuracy([_loc("a.py", 1, 2)], [])["precision"] == 0.0


def test_file_level_exact_prediction():
    scores = evaluate_file_level_accuracy([_loc("a.py", 1, 2)], [_loc("a.py", 5, 9)])
    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["f1"] == 1.0
    assert scores["exact_match"] == 1.0


def test_file_level_accuracy_is_jaccard_not_recall():
    scores = evaluate_file_level_accuracy(
        [_loc("a.py", 1, 2), _loc("b.py", 1, 2)], [_loc("a.py", 1, 2), _loc("c.py", 1, 2)]
    )
    assert scores["precision"] == 0.5
    assert scores["recall"] == 0.5
    assert scores["exact_match"] == 0.0
    assert scores["accuracy"] == pytest.approx(1 / 3)


def test_file_level_superset_prediction():
    scores = evaluate_file_level_accuracy(
        [_loc("a.py", 1, 2), _loc("b.py", 1, 2)],
        [_loc("a.py", 1, 2), _loc("b.py", 1, 2), _loc("c.py", 1, 2)],
    )
    assert scores["precision"] == pytest.approx(2 / 3)
    assert scores["recall"] == 1.0
    assert scores["exact_match"] == 0.0


def test_file_level_ignores_locations_without_file_path():
    scores = evaluate_file_level_accuracy([_loc("a.py", 1, 2)], [{"raw": "unparsable"}])
    assert scores["precision"] == 0.0
    assert scores["recall"] == 0.0


def test_chunk_containment_exact_match():
    scores = evaluate_chunk_containment_metrics([_loc("a.py", 10, 20)], [_loc("a.py", 10, 20)])
    assert scores["coverage_recall"] == 1.0
    assert scores["avg_prediction_tightness"] == 1.0
    assert scores["precision"] == 1.0


def test_chunk_containment_requires_same_file():
    scores = evaluate_chunk_containment_metrics([_loc("a.py", 10, 20)], [_loc("b.py", 10, 20)])
    assert scores["coverage_recall"] == 0.0
    assert scores["total_chunks"] == 1
    assert scores["total_predictions"] == 1


def test_chunk_containment_wide_prediction_lowers_tightness():
    scores = evaluate_chunk_containment_metrics([_loc("a.py", 10, 20)], [_loc("a.py", 5, 30)])
    assert scores["coverage_recall"] == 1.0
    assert scores["avg_prediction_tightness"] == pytest.approx(11 / 26)


def test_chunk_containment_is_strict_not_overlap():
    scores = evaluate_chunk_containment_metrics([_loc("a.py", 10, 20)], [_loc("a.py", 11, 20)])
    assert scores["coverage_recall"] == 0.0


def test_chunk_containment_no_predictions():
    scores = evaluate_chunk_containment_metrics([_loc("a.py", 10, 20)], [])
    assert scores["coverage_recall"] == 0.0
    assert scores["total_chunks"] == 1


def test_chunk_tightness_is_conditioned_on_covered_chunks():
    gold = [_loc("a.py", 10, 20), _loc("a.py", 50, 60), _loc("a.py", 80, 90)]
    scores = evaluate_chunk_containment_metrics(gold, [_loc("a.py", 10, 20)])
    assert scores["coverage_recall"] == pytest.approx(1 / 3)
    assert scores["avg_prediction_tightness"] == 1.0
