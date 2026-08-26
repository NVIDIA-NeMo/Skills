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

from recipes.sherloc.metrics.sherloc import SherlocMetrics


def _make_prediction(score, gt_count=2, pred_count=2, status="success"):
    """Build one generation whose evaluator scores are all `score`."""
    return {
        "status": status,
        "eval_status": [
            {
                "file_level": {
                    "precision": score,
                    "recall": score,
                    "f1": score,
                    "exact_match": score,
                    "accuracy": score,
                },
                "chunk_containment": {
                    "coverage_recall": score,
                    "avg_prediction_tightness": score,
                    "precision": score,
                },
                "ground_truth_count": gt_count,
                "predicted_count": pred_count,
            }
        ],
    }


def test_metrics_average_two_instances():
    metrics = SherlocMetrics()
    metrics.update([_make_prediction(1.0)])
    metrics.update([_make_prediction(0.0)])
    result = metrics.get_metrics()["pass@1"]
    assert result["file_precision"] == 50.0
    assert result["file_recall"] == 50.0
    assert result["file_f1"] == 50.0
    assert result["file_accuracy"] == 50.0
    assert result["chunk_coverage_recall"] == 50.0


def test_metrics_scale_to_percent_once():
    metrics = SherlocMetrics()
    metrics.update([_make_prediction(1.0)])
    assert metrics.get_metrics()["pass@1"]["file_recall"] == 100.0


def test_metrics_count_locations():
    metrics = SherlocMetrics()
    metrics.update([_make_prediction(1.0, gt_count=2, pred_count=3)])
    result = metrics.get_metrics()["pass@1"]
    assert result["total_gt_locs"] == 2
    assert result["total_pred_locs"] == 3
    assert result["num_entries"] == 1


@pytest.mark.parametrize(
    "prediction",
    [{"eval_status": []}, {}, {"eval_status": ["not a dict"]}, {"eval_status": [{}]}],
)
def test_malformed_eval_status_scores_zero(prediction):
    scores = SherlocMetrics()._get_score_dict(prediction)
    assert set(scores) == {
        "file_precision",
        "file_recall",
        "file_f1",
        "file_exact_match",
        "file_accuracy",
        "chunk_coverage_recall",
        "chunk_avg_tightness",
        "chunk_precision",
    }
    assert all(value == 0.0 for value in scores.values())


def test_instances_without_ground_truth_are_excluded():
    metrics = SherlocMetrics()
    metrics.update([_make_prediction(1.0, gt_count=2)])
    metrics.update([_make_prediction(0.0, gt_count=0)])
    result = metrics.get_metrics()["pass@1"]
    assert result["samples_no_gt"] == 1
    assert result["samples_with_gt"] == 1
    assert result["file_recall"] == 100.0


def test_reset_clears_state():
    metrics = SherlocMetrics()
    metrics.update([_make_prediction(1.0)])
    metrics.reset()
    assert metrics.total == 0
    assert metrics.file_level_metrics == []
    assert metrics.chunk_containment_metrics == []
    assert metrics.total_gt_locations == 0
    assert metrics.total_pred_locations == 0
    assert metrics.successful_samples == 0
    assert metrics.failed_samples == 0
    assert metrics.skipped_samples == 0
    assert metrics.no_gt_samples == 0
    assert metrics.get_metrics() == {}


def test_get_incorrect_sample_scores_zero():
    metrics = SherlocMetrics()
    incorrect = SherlocMetrics.get_incorrect_sample(_make_prediction(1.0, gt_count=2))
    metrics.update([incorrect])
    result = metrics.get_metrics()["pass@1"]
    assert result["samples_no_gt"] == 0
    assert result["samples_with_gt"] == 1
    assert result["total_gt_locs"] == 2
    assert result["file_recall"] == 0.0
    assert result["chunk_coverage_recall"] == 0.0


def test_metrics_to_print_keys():
    assert set(SherlocMetrics().metrics_to_print()) == {
        "num_entries",
        "samples_with_gt",
        "samples_no_gt",
        "total_gt_locs",
        "total_pred_locs",
        "file_precision",
        "file_recall",
        "file_f1",
        "file_accuracy",
        "chunk_coverage_recall",
        "chunk_avg_tightness",
        "chunk_precision",
    }


def test_aggregate_of_empty_list_is_empty():
    assert SherlocMetrics()._compute_aggregate_metrics([]) == {}


def test_processing_stats_are_the_only_source_for_run_counters():
    successful = _make_prediction(1.0)
    successful["processing_stats"] = {
        "total_samples": 2,
        "successful_samples": 1,
        "failed_samples": 0,
        "skipped_samples": 1,
    }
    skipped = _make_prediction(0.0, status="skipped")
    skipped["eval_status"][0]["is_skipped_sample"] = True

    metrics = SherlocMetrics()
    metrics.update([successful])
    metrics.update([skipped])
    result = metrics.get_metrics()["pass@1"]

    assert result["successful_samples"] == 1
    assert result["failed_samples"] == 0
    assert result["skipped_samples"] == 1
    assert result["success_rate"] == 50.0
