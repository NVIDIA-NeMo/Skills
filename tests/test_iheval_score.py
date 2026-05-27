# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Tests for the IHEval group-level composite score (compute_score)."""

from nemo_skills.dataset.iheval.iheval_score import compute_score


def _sub(entries, symbolic, **extra):
    return {"pass@1": {"num_entries": entries, "symbolic_correct": symbolic, **extra}}


def _full_metrics(**overrides):
    """Construct metrics for all 9 subs, each with a default or overridden value."""
    defaults = {
        "rule_following_single": (100, 50.0),
        "rule_following_multi": (100, 40.0),
        "task_execution_verb_extract": (250, 60.0),
        "task_execution_translation": (250, 55.0),
        "task_execution_lang_detect": (240, 70.0),
        "safety_hijack": (100, 80.0),
        "safety_extract": (100, 75.0),
        "tool_use_webpage": (150, 65.0),
        "tool_use_slack_user": (100, 45.0),
    }
    metrics = {}
    for sub, (n, s) in defaults.items():
        extra = overrides.pop(sub, {})
        metrics[f"iheval.{sub}"] = _sub(n, s, **extra)
    assert not overrides, f"unused overrides: {overrides}"
    return metrics


class TestComputeScore:
    def test_shape(self):
        out = compute_score(_full_metrics())
        assert list(out.keys()) == ["iheval"]
        pass1 = out["iheval"]["pass@1"]
        expected_keys = {
            "num_entries",
            "symbolic_correct",
            "aligned_avg",
            "conflict_avg",
            "reference_avg",
            "conflict_gap",
            "rule_following_avg",
            "task_execution_avg",
            "safety_avg",
            "tool_use_avg",
        }
        assert set(pass1.keys()) == expected_keys

    def test_overall_is_mean_of_subs(self):
        out = compute_score(_full_metrics())
        expected = (50 + 40 + 60 + 55 + 70 + 80 + 75 + 65 + 45) / 9
        assert abs(out["iheval"]["pass@1"]["symbolic_correct"] - expected) < 1e-9

    def test_category_averages(self):
        pass1 = compute_score(_full_metrics())["iheval"]["pass@1"]
        assert pass1["rule_following_avg"] == (50 + 40) / 2
        assert pass1["task_execution_avg"] == (60 + 55 + 70) / 3
        assert pass1["safety_avg"] == (80 + 75) / 2
        assert pass1["tool_use_avg"] == (65 + 45) / 2

    def test_total_entries_sum(self):
        out = compute_score(_full_metrics())
        assert out["iheval"]["pass@1"]["num_entries"] == 100 + 100 + 250 + 250 + 240 + 100 + 100 + 150 + 100

    def test_per_setting_avgs_and_gap(self):
        metrics = _full_metrics(
            rule_following_single={"setting_aligned": 60.0, "setting_conflict": 30.0, "setting_reference": 70.0},
            rule_following_multi={"setting_aligned": 50.0, "setting_conflict": 20.0, "setting_reference": 60.0},
            safety_hijack={"setting_aligned": 90.0, "setting_conflict": 70.0},  # no reference
            safety_extract={"setting_aligned": 85.0, "setting_conflict": 65.0},
        )
        out = compute_score(metrics)["iheval"]["pass@1"]
        assert out["aligned_avg"] == (60 + 50 + 90 + 85) / 4
        assert out["conflict_avg"] == (30 + 20 + 70 + 65) / 4
        assert out["reference_avg"] == (70 + 60) / 2
        assert out["conflict_gap"] == out["reference_avg"] - out["conflict_avg"]

    def test_missing_sub_treated_as_skipped(self):
        metrics = _full_metrics()
        del metrics["iheval.safety_hijack"]  # simulate failed sub
        pass1 = compute_score(metrics)["iheval"]["pass@1"]
        expected_overall = (50 + 40 + 60 + 55 + 70 + 75 + 65 + 45) / 8
        assert abs(pass1["symbolic_correct"] - expected_overall) < 1e-9
        assert pass1["safety_avg"] == 75.0

    def test_empty_metrics(self):
        pass1 = compute_score({})["iheval"]["pass@1"]
        assert pass1["num_entries"] == 0
        assert pass1["symbolic_correct"] == 0.0
        assert pass1["conflict_gap"] == 0.0

    def test_symbolic_correct_is_representative_key(self):
        # aggregate_metrics picks `symbolic_correct` as the representative pass@1 value.
        pass1 = compute_score(_full_metrics())["iheval"]["pass@1"]
        assert "symbolic_correct" in pass1
        assert isinstance(pass1["symbolic_correct"], float)

    def test_pass_at_k_modes_are_aggregated(self):
        # iheval:N -> each sub reports extra agg modes; the group composite must surface them all.
        metrics = _full_metrics()
        for name, sub in metrics.items():
            base = sub["pass@1"]["symbolic_correct"]
            sub["pass@4"] = {"num_entries": sub["pass@1"]["num_entries"], "symbolic_correct": base + 10}
            sub["pass@1[avg-of-4]"] = {"num_entries": sub["pass@1"]["num_entries"], "symbolic_correct": base - 5}
        out = compute_score(metrics)["iheval"]
        assert set(out.keys()) == {"pass@1", "pass@4", "pass@1[avg-of-4]"}
        base_overall = (50 + 40 + 60 + 55 + 70 + 80 + 75 + 65 + 45) / 9
        assert out["pass@1"]["symbolic_correct"] == base_overall
        assert out["pass@4"]["symbolic_correct"] == base_overall + 10
        assert out["pass@1[avg-of-4]"]["symbolic_correct"] == base_overall - 5
