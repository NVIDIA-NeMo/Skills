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
                    "correct_std_dev_across_runs": 0.0,
                    "correct_avg_sample_std_dev": 0.7071067811865476,
                    "correct_std_err_across_runs": 0.0,
                }
            },
        ),
        (
            3,
            [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]],
            {
                "pass@1[avg-of-2]": {
                    "correct_std_dev_across_runs": 0.0,
                    "correct_avg_sample_std_dev": 0.47140452079103173,
                    "correct_std_err_across_runs": 0.0,
                },
                "pass@1[avg-of-3]": {
                    "correct_std_dev_across_runs": 0.0,
                    "correct_avg_sample_std_dev": 0.3849001794597506,
                    "correct_std_err_across_runs": 0.0,
                },
            },
        ),
        (
            4,
            [[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0]],
            {
                "pass@1[avg-of-2]": {
                    "correct_std_dev_across_runs": 0.0,
                    "correct_avg_sample_std_dev": 0.47140452079103173,
                    "correct_std_err_across_runs": 0.0,
                },
                "pass@1[avg-of-3]": {
                    "correct_std_dev_across_runs": 0.0,
                    "correct_avg_sample_std_dev": 0.5773502691896258,
                    "correct_std_err_across_runs": 0.0,
                },
                "pass@1[avg-of-4]": {
                    "correct_std_dev_across_runs": 0.16666666666666666,
                    "correct_avg_sample_std_dev": 0.5515668461264172,
                    "correct_std_err_across_runs": 0.08333333333333333,
                },
            },
        ),
        (
            5,
            [
                [1.0, 0.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0],
            ],
            {
                "pass@1[avg-of-2]": {
                    "correct_std_dev_across_runs": 0.0,
                    "correct_avg_sample_std_dev": 0.3535533905932738,
                    "correct_std_err_across_runs": 0.0,
                },
                "pass@1[avg-of-3]": {
                    "correct_std_dev_across_runs": 0.14433756729740646,
                    "correct_avg_sample_std_dev": 0.4330127018922194,
                    "correct_std_err_across_runs": 0.08333333333333334,
                },
                "pass@1[avg-of-4]": {
                    "correct_std_dev_across_runs": 0.14433756729740643,
                    "correct_avg_sample_std_dev": 0.5386751345948129,
                    "correct_std_err_across_runs": 0.07216878364870322,
                },
                "pass@1[avg-of-5]": {
                    "correct_std_dev_across_runs": 0.13693063937629152,
                    "correct_avg_sample_std_dev": 0.5477225575051662,
                    "correct_std_err_across_runs": 0.06123724356957944,
                },
            },
        ),
        (
            2,
            [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0]],
            {
                "pass@1[avg-of-2]": {
                    "correct_std_dev_across_runs": 0.0,
                    "correct_avg_sample_std_dev": 0.0,
                    "correct_std_err_across_runs": 0.0,
                }
            },
        ),
    ],
)
def test_add_std_metrics(
    max_k: int, scores_list: list[list[bool | int | float]], expected_result: dict[str, dict[str, float]]
) -> None:
    metrics = MockMetrics()
    metrics.max_k = max_k
    metrics.all_scores = {"correct": scores_list}
    metrics_dict: dict[str, dict[str, float]] = {}
    for i in range(2, max_k + 1):
        metrics_dict[f"pass@1[avg-of-{i}]"] = {}
    metrics._add_std_metrics(metrics_dict)
    for eval_mode, expected_values in expected_result.items():
        for metric_name, expected_value in expected_values.items():
            actual_value = metrics_dict[eval_mode][metric_name]
            assert abs(actual_value - expected_value) < 1e-10
