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

from nemo_skills.evaluation.metrics.base import BaseMetrics


class MockMetrics(BaseMetrics):
    def _get_score_dict(self, prediction):
        return {"correct": float(prediction.get("is_correct", False))}


@pytest.mark.parametrize(
    "max_k,scores_list,expected_result",
    [
        (1, [[1.0]], {}),
        (
            2,
            [[1.0, 0.0], [0.0, 1.0]],
            {
                "pass@1[avg-of-2]": {
                    "correct_statistics": {
                        "avg": 0.5,
                        "std_dev_across_runs": 0.0,
                        "avg_sample_std_dev": 0.7071067811865476,
                        "std_err_across_runs": 0.0,
                    }
                }
            },
        ),
        (
            3,
            [[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]],
            {
                "pass@1[avg-of-2]": {
                    "correct_statistics": {
                        "avg": 0.6666666666666666,
                        "std_dev_across_runs": 0.0,
                        "avg_sample_std_dev": 0.47140452079103173,
                        "std_err_across_runs": 0.0,
                    }
                },
                "pass@1[avg-of-3]": {
                    "correct_statistics": {
                        "avg": 0.6666666666666666,
                        "std_dev_across_runs": 0.0,
                        "avg_sample_std_dev": 0.3849001794597506,
                        "std_err_across_runs": 0.0,
                    }
                },
            },
        ),
        (
            4,
            [[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            {
                "pass@1[avg-of-2]": {
                    "correct_statistics": {
                        "avg": 0.6666666666666666,
                        "std_dev_across_runs": 0.0,
                        "avg_sample_std_dev": 0.47140452079103173,
                        "std_err_across_runs": 0.0,
                    }
                },
                "pass@1[avg-of-3]": {
                    "correct_statistics": {
                        "avg": 0.6666666666666666,
                        "std_dev_across_runs": 0.0,
                        "avg_sample_std_dev": 0.5773502691896258,
                        "std_err_across_runs": 0.0,
                    }
                },
                "pass@1[avg-of-4]": {
                    "correct_statistics": {
                        "avg": 0.5833333333333334,
                        "std_dev_across_runs": 0.16666666666666666,
                        "avg_sample_std_dev": 0.5515668461264172,
                        "std_err_across_runs": 0.08333333333333333,
                    }
                },
            },
        ),
        (
            5,
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
            ],
            {
                "pass@1[avg-of-2]": {
                    "correct_statistics": {
                        "avg": 0.5,
                        "std_dev_across_runs": 0.0,
                        "avg_sample_std_dev": 0.3535533905932738,
                        "std_err_across_runs": 0.0,
                    }
                },
                "pass@1[avg-of-3]": {
                    "correct_statistics": {
                        "avg": 0.5833333333333334,
                        "std_dev_across_runs": 0.14433756729740646,
                        "avg_sample_std_dev": 0.4330127018922194,
                        "std_err_across_runs": 0.08333333333333334,
                    }
                },
                "pass@1[avg-of-4]": {
                    "correct_statistics": {
                        "avg": 0.625,
                        "std_dev_across_runs": 0.14433756729740643,
                        "avg_sample_std_dev": 0.5386751345948129,
                        "std_err_across_runs": 0.07216878364870322,
                    }
                },
                "pass@1[avg-of-5]": {
                    "correct_statistics": {
                        "avg": 0.6,
                        "std_dev_across_runs": 0.13693063937629152,
                        "avg_sample_std_dev": 0.5477225575051662,
                        "std_err_across_runs": 0.06123724356957944,
                    }
                },
            },
        ),
        (
            2,
            [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            {
                "pass@1[avg-of-2]": {
                    "correct_statistics": {
                        "avg": 0.6666666666666666,
                        "std_dev_across_runs": 0.0,
                        "avg_sample_std_dev": 0.0,
                        "std_err_across_runs": 0.0,
                    }
                }
            },
        ),
    ],
)
def test_base_metrics_add_std_metrics(
    max_k: int, scores_list: list[list[bool | int | float]], expected_result: dict[str, dict[str, float]]
) -> None:
    metrics = MockMetrics()
    metrics.max_k = max_k
    metrics.all_scores = {"correct": scores_list}
    metrics_dict: dict[str, dict[str, float]] = {}
    for i in range(2, max_k + 1):
        metrics_dict[f"pass@1[avg-of-{i}]"] = {}
    metrics._add_std_metrics(metrics_dict)
    assert metrics_dict == expected_result


@pytest.mark.parametrize(
    "predictions,expected_all_scores",
    [
        (
            [
                {"num_reasoning_tokens": 80, "num_answer_tokens": 20, "is_correct": True},
                {"num_reasoning_tokens": 90, "num_answer_tokens": 30, "is_correct": False},
            ],
            {"reasoning_tokens": [[80], [90]], "answer_tokens": [[20], [30]]},
        ),
        (
            [{"num_generated_tokens": 50, "is_correct": True}, {"num_generated_tokens": 60, "is_correct": False}],
            {"reasoning_tokens": [[0], [0]], "answer_tokens": [[50], [60]]},
        ),
        (
            [
                {"num_reasoning_tokens": 100, "num_answer_tokens": 40, "is_correct": True},
                {"num_generated_tokens": 80, "is_correct": False},
            ],
            {"reasoning_tokens": [[100], [0]], "answer_tokens": [[40], [80]]},
        ),
    ],
)
def test_base_metrics_update(predictions, expected_all_scores):
    """Test the base update method's token handling (scores are handled by subclasses)."""
    metrics = MockMetrics()
    metrics.update(predictions)
    assert metrics.all_scores == expected_all_scores
