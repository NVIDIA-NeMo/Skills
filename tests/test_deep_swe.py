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

from nemo_skills.evaluation.metrics.code_metrics import DeepSweMetrics
from nemo_skills.inference.eval.deepswe import parse_deepswe_reward


def test_parse_deepswe_reward_no_patch():
    assert parse_deepswe_reward(None, patch_exists=False) == {
        "resolved": False,
        "patch_exists": False,
        "patch_successfully_applied": False,
        "reward": 0,
        "f2p": 0.0,
        "p2p": 0.0,
        "partial": 0.0,
    }


def test_parse_deepswe_reward_apply_failed():
    metrics = parse_deepswe_reward({"reward": 0, "apply_failed": 1, "f2p": 0.0, "p2p": 1.0}, patch_exists=True)
    assert metrics["resolved"] is False
    assert metrics["patch_successfully_applied"] is False


def test_parse_deepswe_reward_success():
    metrics = parse_deepswe_reward(
        {
            "reward": 1,
            "f2p": 1.0,
            "p2p": 1.0,
            "partial": 1.0,
            "f2p_passed": 3,
            "f2p_total": 3,
            "p2p_passed": 2,
            "p2p_total": 2,
        },
        patch_exists=True,
    )
    assert metrics["resolved"] is True
    assert metrics["patch_exists"] is True
    assert metrics["patch_successfully_applied"] is True
    assert metrics["reward"] == 1.0
    assert metrics["f2p"] == 1.0


def test_deepswe_metrics_aggregate():
    metrics = DeepSweMetrics()
    metrics.update(
        [
            {
                "deep-swe-metrics": {
                    "resolved": True,
                    "patch_exists": True,
                    "patch_successfully_applied": True,
                    "reward": 1,
                    "f2p": 1.0,
                    "p2p": 1.0,
                    "partial": 1.0,
                }
            }
        ]
    )
    metrics.update(
        [
            {
                "deep-swe-metrics": {
                    "resolved": False,
                    "patch_exists": True,
                    "patch_successfully_applied": True,
                    "reward": 0,
                    "f2p": 0.5,
                    "p2p": 1.0,
                    "partial": 0.75,
                }
            }
        ]
    )
    # BaseMetrics stores means over updates; spot-check resolved rate semantics via score dict helper.
    score = metrics._get_score_dict(
        {
            "deep-swe-metrics": {
                "resolved": True,
                "patch_exists": True,
                "patch_successfully_applied": True,
                "reward": 1,
                "f2p": 1.0,
                "p2p": 1.0,
                "partial": 1.0,
            }
        }
    )
    assert score["issues_resolved"] is True
    assert score["reward"] == 1.0
