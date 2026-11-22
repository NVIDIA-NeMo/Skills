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

import json

import pytest

from nemo_skills.evaluation.metrics.code_metrics import (
    BigCodeBenchMetrics,
    ComputeEvalMetrics,
    EvalPlusMetrics,
    HumanEvalInfillingMetrics,
    LiveCodeBenchMetrics,
    SweBenchMetrics,
)
from nemo_skills.evaluation.metrics.compute_metrics import ComputeMetrics


@pytest.mark.parametrize(
    "metrics_cls,correct_fields",
    [
        (EvalPlusMetrics, {"is_correct": True, "is_correct-plus": True}),
        (LiveCodeBenchMetrics, {"graded_list": [True]}),
        (
            SweBenchMetrics,
            {"swe-bench-metrics": {"resolved": True, "patch_exists": True, "patch_successfully_applied": True}},
        ),
        (BigCodeBenchMetrics, {"status": "pass"}),
        (HumanEvalInfillingMetrics, {"passed": True}),
        (ComputeEvalMetrics, {"passed": True}),
    ],
)
def test_get_incorrect_sample_keeps_other_fields(metrics_cls, correct_fields):
    metrics = metrics_cls()
    prediction = {"num_generated_tokens": 5000, "subset_for_metrics": "hard", **correct_fields}

    incorrect = metrics.get_incorrect_sample(prediction)

    assert incorrect["num_generated_tokens"] == 5000
    assert incorrect["subset_for_metrics"] == "hard"
    # the first score is the main correctness score of each metric
    assert not next(iter(metrics._get_score_dict(incorrect).values()))
    # the input prediction is left untouched
    assert prediction == {"num_generated_tokens": 5000, "subset_for_metrics": "hard", **correct_fields}


def test_max_seq_len_keeps_subset_and_tokens(tmp_path):
    rows = [
        {"graded_list": [True], "num_generated_tokens": 5000, "subset_for_metrics": "hard"},
        {"graded_list": [True], "num_generated_tokens": 100, "subset_for_metrics": "easy"},
        {"graded_list": [False], "num_generated_tokens": 100, "subset_for_metrics": "hard"},
    ]
    output_file = tmp_path / "output.jsonl"
    output_file.write_text("".join(json.dumps(row) + "\n" for row in rows))

    metrics = ComputeMetrics(benchmark="livecodebench", metric_type="livecodebench", max_seq_len=1000).compute_metrics(
        [str(output_file)]
    )

    assert metrics["_all_"]["pass@1"]["num_entries"] == 3
    assert metrics["_all_"]["pass@1"]["avg_tokens"] == 1733
    assert metrics["hard"]["pass@1"]["num_entries"] == 2
    assert metrics["hard"]["pass@1"]["accuracy"] == 0.0
    assert metrics["easy"]["pass@1"]["num_entries"] == 1
