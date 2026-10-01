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

from nemo_skills.evaluation.metrics.facts_grounding_metrics import FactsGroundingMetrics


def test_facts_grounding_eligibility_consensus_disqualifies_only_unanimous_ineligible():
    metrics = FactsGroundingMetrics()

    partially_eligible = {
        "judgement_grounding_per_judge": {"judge_a": True, "judge_b": False, "judge_c": True},
        "judgement_quality_per_judge": {"judge_a": False, "judge_b": True, "judge_c": False},
    }
    unanimously_ineligible = {
        "judgement_grounding_per_judge": {"judge_a": True, "judge_b": True, "judge_c": True},
        "judgement_quality_per_judge": {"judge_a": False, "judge_b": False, "judge_c": False},
    }

    partial_scores = metrics._get_score_dict(partially_eligible)
    assert partial_scores["eligibility_rate"] is True
    assert partial_scores["final_factuality"] == pytest.approx(2 / 3)

    ineligible_scores = metrics._get_score_dict(unanimously_ineligible)
    assert ineligible_scores["eligibility_rate"] is False
    assert ineligible_scores["final_factuality"] == 0.0

    metrics.update([partially_eligible])
    metrics.update([unanimously_ineligible])
    aggregated = metrics.get_metrics()["pass@1"]

    assert aggregated["eligibility_rate"] == 50.0
    assert aggregated["unadjusted_factuality"] == pytest.approx(100.0 * ((2 / 3) + 1.0) / 2)
    assert aggregated["final_factuality"] == pytest.approx(100.0 * (2 / 3) / 2)
