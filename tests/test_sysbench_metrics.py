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

from nemo_skills.evaluation.metrics.sysbench_metrics import SysBenchMetrics


def test_parse_judgement():
    """Test parsing JSON judgement with unquoted integer keys."""
    judgement = '{"评判结果": {1: "是", 2: "否"}}'
    result = SysBenchMetrics.parse_sysbench_judgement(judgement)
    assert result is not None
    assert result["评判结果"]["1"] == "是"


def test_is_correct():
    """Test correctness check."""
    assert SysBenchMetrics.is_sysbench_correct('{"评判结果": {"1": "是", "2": "是"}}') is True
    assert SysBenchMetrics.is_sysbench_correct('{"评判结果": {"1": "是", "2": "否"}}') is False
    assert SysBenchMetrics.is_sysbench_correct("INVALID") is False
