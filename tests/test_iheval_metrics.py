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

"""Tests for IHEval metrics aggregation (per-setting / per-variant on top of pass@k)."""

import pytest

from nemo_skills.evaluation.metrics.iheval_metrics import IHEvalMetrics


def _row(correct, setting=None, variant=None, tokens=10):
    r = {"symbolic_correct": correct, "num_generated_tokens": tokens}
    if setting is not None:
        r["setting"] = setting
    if variant is not None:
        r["variant"] = variant
    return r


def _update_all(m, rows):
    for row in rows:
        m.update([row])


class TestIHEvalMetricsAllCorrect:
    def test_all_correct_reports_100_everywhere(self):
        m = IHEvalMetrics()
        rows = [
            _row(1.0, "aligned", "default"),
            _row(1.0, "aligned", "strong"),
            _row(1.0, "conflict", "default"),
            _row(1.0, "conflict", "strong"),
            _row(1.0, "reference", "default"),
        ]
        _update_all(m, rows)
        out = m.get_metrics()
        assert "pass@1" in out
        agg = out["pass@1"]
        assert agg["symbolic_correct"] == pytest.approx(100.0)
        assert agg["setting_aligned"] == pytest.approx(100.0)
        assert agg["setting_conflict"] == pytest.approx(100.0)
        assert agg["setting_reference"] == pytest.approx(100.0)
        assert agg["variant_default"] == pytest.approx(100.0)
        assert agg["variant_strong"] == pytest.approx(100.0)
        assert agg["num_entries"] == 5


class TestIHEvalMetricsMixed:
    def test_mixed_breakdown(self):
        m = IHEvalMetrics()
        rows = []
        for _ in range(3):
            rows.append(_row(1.0, "aligned", "default"))
        for _ in range(2):
            rows.append(_row(0.0, "aligned", "default"))
        rows.append(_row(1.0, "conflict", "strong"))
        for _ in range(4):
            rows.append(_row(0.0, "conflict", "strong"))
        for _ in range(3):
            rows.append(_row(1.0, "reference", "default"))
        _update_all(m, rows)
        agg = m.get_metrics()["pass@1"]
        assert agg["symbolic_correct"] == pytest.approx(100.0 * 7 / 13)
        assert agg["setting_aligned"] == pytest.approx(60.0)
        assert agg["setting_conflict"] == pytest.approx(20.0)
        assert agg["setting_reference"] == pytest.approx(100.0)
        assert agg["variant_default"] == pytest.approx(75.0)
        assert agg["variant_strong"] == pytest.approx(20.0)
        assert agg["num_entries"] == 13


class TestIHEvalMetricsMissingKeys:
    def test_no_reference_rows_omits_reference_key(self):
        m = IHEvalMetrics()
        _update_all(m, [_row(1.0, "aligned", "default"), _row(0.0, "conflict", "default")])
        agg = m.get_metrics()["pass@1"]
        assert "setting_aligned" in agg
        assert "setting_conflict" in agg
        assert "setting_reference" not in agg

    def test_missing_setting_key_omits_setting_breakdown(self):
        m = IHEvalMetrics()
        _update_all(m, [_row(1.0, variant="default"), _row(0.0, variant="strong")])
        agg = m.get_metrics()["pass@1"]
        assert not any(k.startswith("setting_") for k in agg)
        assert "variant_default" in agg
        assert "variant_strong" in agg

    def test_missing_variant_key_omits_variant_breakdown(self):
        m = IHEvalMetrics()
        _update_all(m, [_row(1.0, setting="aligned"), _row(0.0, setting="conflict")])
        agg = m.get_metrics()["pass@1"]
        assert not any(k.startswith("variant_") for k in agg)
        assert "setting_aligned" in agg
        assert "setting_conflict" in agg

    def test_all_missing_still_tracks_overall(self):
        m = IHEvalMetrics()
        _update_all(m, [_row(1.0), _row(0.0), _row(1.0)])
        agg = m.get_metrics()["pass@1"]
        assert agg["symbolic_correct"] == pytest.approx(100.0 * 2 / 3)
        assert not any(k.startswith("setting_") for k in agg)
        assert not any(k.startswith("variant_") for k in agg)
        assert agg["num_entries"] == 3


class TestIHEvalMetricsFloatScores:
    def test_fractional_symbolic_correct(self):
        m = IHEvalMetrics()
        _update_all(
            m,
            [
                _row(0.5, "aligned", "default"),
                _row(0.25, "aligned", "default"),
                _row(1.0, "conflict", "strong"),
            ],
        )
        agg = m.get_metrics()["pass@1"]
        assert agg["symbolic_correct"] == pytest.approx(100.0 * 1.75 / 3)
        assert agg["setting_aligned"] == pytest.approx(37.5)
        assert agg["setting_conflict"] == pytest.approx(100.0)


class TestIHEvalMetricsPassAtK:
    def test_iheval_n_yields_pass_at_1_avg_of_n(self):
        # iheval:N feeds N predictions per question; metrics should expose pass@1[avg-of-N].
        m = IHEvalMetrics()
        # one question, 2 samples (one correct, one wrong) -> avg-of-2 = 50%
        m.update([_row(1.0, "aligned", "default"), _row(0.0, "aligned", "default")])
        out = m.get_metrics()
        assert "pass@1[avg-of-2]" in out
        assert out["pass@1[avg-of-2]"]["symbolic_correct"] == pytest.approx(50.0)
        # per-setting breakdown uses avg-over-attempts -> also 50%
        assert out["pass@1[avg-of-2]"]["setting_aligned"] == pytest.approx(50.0)


class TestIHEvalMetricsReset:
    def test_reset_clears_all_state(self):
        m = IHEvalMetrics()
        _update_all(m, [_row(1.0, "aligned", "default"), _row(0.0, "conflict", "strong")])
        assert m.total == 2
        m.reset()
        assert m.total == 0
        assert m._setting_totals == {}
        assert m._variant_totals == {}
        assert len(m.eval_dict) == 0
