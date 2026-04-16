# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import math
from collections import defaultdict

from nemo_skills.evaluation.metrics.base import BaseMetrics

_SENT_LABELS = ("supported", "unsupported", "contradictory", "no_rad")
_WILSON_Z_95 = 1.96


def _wilson_half_width_pct(percentage_value: float, n: int) -> float:
    """Normal-approx half-width of a 95% CI on a proportion (in percentage units)."""
    if n <= 0:
        return 0.0
    p = max(0.0, min(1.0, percentage_value / 100.0))
    return 100.0 * _WILSON_Z_95 * math.sqrt(p * (1.0 - p) / n)


class FactsGroundingMetrics(BaseMetrics):
    """Metrics for the Google FACTS Grounding benchmark.

    Mirrors the aggregation in the google-facts reference ``compute_scores``:

    - ``unadjusted_factuality``: mean over samples of the per-sample judge mean
      (i.e. fraction of judge-votes that mark the response grounded).
      Equals the reference's ``average_grounding_score`` and matches the paper's
      Table 5 "unadjusted factuality" when averaged across judges.
    - ``final_factuality``: same, but per-sample grounding is zeroed when the
      sample fails the consensus eligibility check (``all`` judges eligible).
      Equals the reference's ``average_grounding_score_where_quality_check_passed``
      and matches paper Table 6.
    - ``eligibility_rate``: fraction of samples where ``all`` judges passed
      the ineligible-responses filter (reference's ``average_quality_check_passed``).
    - ``unadjusted_{judge}`` / ``eligibility_{judge}``: per-judge slices
      (reference's ``average_grounding_scores_per_model`` + a matching quality row).

    Sentence-level breakdown (JSON-method judges only, micro-averaged across
    all (sample, judge) pairs that produced parseable output):
    - ``pct_supported`` / ``pct_unsupported`` / ``pct_contradictory`` / ``pct_no_rad``
    - ``avg_sentences`` (avg per parsed judgement)
    - ``sentence_stats_coverage`` (share of (sample, judge) pairs that parsed)

    Confidence intervals are 95% normal-approximation half-widths, reported as
    ``*_ci95`` in percentage units.

    Backward-compat fields (matching the single-judge implementation that
    preceded this one) are preserved:
    - ``grounding_correct`` (unanimous: all judges say grounded)
    - ``quality_passed`` (alias of ``eligibility_rate``)
    - ``factuality_correct`` (unanimous AND eligible)
    """

    def __init__(self, compute_no_answer: bool = False):
        super().__init__(compute_no_answer=compute_no_answer)

    def reset(self):
        super().reset()
        self._judge_grounding_correct: dict[str, int] = defaultdict(int)
        self._judge_grounding_total: dict[str, int] = defaultdict(int)
        self._judge_quality_correct: dict[str, int] = defaultdict(int)
        self._judge_quality_total: dict[str, int] = defaultdict(int)
        self._sentence_totals: dict[str, int] = {label: 0 for label in _SENT_LABELS}
        self._sentence_totals["sentences_total"] = 0
        self._judgements_with_sentence_stats = 0
        self._judgements_with_potential_sentence_stats = 0
        self._judge_tags_seen: list[str] = []

    def _track_judge_tag(self, tag: str) -> None:
        if tag not in self._judge_tags_seen:
            self._judge_tags_seen.append(tag)

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        per_g = prediction.get("judgement_grounding_per_judge") or {}
        per_q = prediction.get("judgement_quality_per_judge") or {}

        if per_g:
            grounding_mean = sum(bool(v) for v in per_g.values()) / len(per_g)
            legacy_grounding = all(bool(v) for v in per_g.values())
        else:
            # Fallback to single-judge field layout.
            legacy_grounding = bool(prediction.get("judgement_grounding", False))
            grounding_mean = float(legacy_grounding)

        if per_q:
            quality_consensus = all(bool(v) for v in per_q.values())
        else:
            quality_consensus = bool(prediction.get("judgement_quality", True))

        final_score = grounding_mean if quality_consensus else 0.0

        return {
            # Paper- and reference-aligned headline metrics.
            "unadjusted_factuality": grounding_mean,
            "eligibility_rate": quality_consensus,
            "final_factuality": final_score,
            # Backward-compat.
            "grounding_correct": legacy_grounding,
            "quality_passed": quality_consensus,
            "factuality_correct": legacy_grounding and quality_consensus,
        }

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions=predictions)

        for pred in predictions:
            per_g = pred.get("judgement_grounding_per_judge") or {}
            per_q = pred.get("judgement_quality_per_judge") or {}
            sent_stats_per_judge = pred.get("sentence_stats_per_judge") or {}

            for tag, verdict in per_g.items():
                self._track_judge_tag(tag)
                self._judge_grounding_total[tag] += 1
                if verdict:
                    self._judge_grounding_correct[tag] += 1

            for tag, verdict in per_q.items():
                self._track_judge_tag(tag)
                self._judge_quality_total[tag] += 1
                if verdict:
                    self._judge_quality_correct[tag] += 1

            for tag, stats in sent_stats_per_judge.items():
                # Only track sentence stats for judges whose method emits them
                # (JSON method); skip implicit-span judges entirely so we don't
                # depress the coverage metric.
                if not stats:
                    continue
                if stats.get("sentence_stats_available"):
                    self._judgements_with_potential_sentence_stats += 1
                    self._judgements_with_sentence_stats += 1
                    self._sentence_totals["sentences_total"] += int(stats.get("sentences_total", 0))
                    for label in _SENT_LABELS:
                        self._sentence_totals[label] += int(stats.get(f"sentences_{label}", 0))
                elif stats.get("sentences_total", 0) == 0 and not stats.get("sentence_stats_available", False):
                    # Judge returned an implicit-span verdict (no sentence stats
                    # ever produced), OR the JSON judge failed to parse.
                    # Distinguish by whether the tag's parser is JSON — we don't
                    # have that context here, so just don't count it against coverage.
                    pass

            # Fallback: legacy single-judge payload with top-level sentence stats.
            if not sent_stats_per_judge and pred.get("sentence_stats_available"):
                self._judgements_with_sentence_stats += 1
                self._judgements_with_potential_sentence_stats += 1
                self._sentence_totals["sentences_total"] += int(pred.get("sentences_total", 0))
                for label in _SENT_LABELS:
                    self._sentence_totals[label] += int(pred.get(f"sentences_{label}", 0))

    def get_metrics(self):
        metrics_dict = super().get_metrics()

        total_sentences = self._sentence_totals["sentences_total"]

        for agg_dict in metrics_dict.values():
            # 95% CIs on the two headline factuality numbers.
            if "unadjusted_factuality" in agg_dict:
                agg_dict["unadjusted_factuality_ci95"] = _wilson_half_width_pct(
                    agg_dict["unadjusted_factuality"], self.total
                )
            if "final_factuality" in agg_dict:
                agg_dict["final_factuality_ci95"] = _wilson_half_width_pct(agg_dict["final_factuality"], self.total)
            if "eligibility_rate" in agg_dict:
                agg_dict["eligibility_rate_ci95"] = _wilson_half_width_pct(agg_dict["eligibility_rate"], self.total)

            # Per-judge breakdown (matches reference's average_grounding_scores_per_model).
            for tag in self._judge_tags_seen:
                g_total = self._judge_grounding_total.get(tag, 0)
                if g_total > 0:
                    rate = 100.0 * self._judge_grounding_correct[tag] / g_total
                    agg_dict[f"unadjusted_{tag}"] = rate
                    agg_dict[f"unadjusted_{tag}_ci95"] = _wilson_half_width_pct(rate, g_total)
                q_total = self._judge_quality_total.get(tag, 0)
                if q_total > 0:
                    rate = 100.0 * self._judge_quality_correct[tag] / q_total
                    agg_dict[f"eligibility_{tag}"] = rate
                    agg_dict[f"eligibility_{tag}_ci95"] = _wilson_half_width_pct(rate, q_total)

            # Sentence-level micro-averages (across parseable (sample, judge) pairs).
            if total_sentences > 0:
                for label in _SENT_LABELS:
                    agg_dict[f"pct_{label}"] = 100.0 * self._sentence_totals[label] / total_sentences
                agg_dict["avg_sentences"] = total_sentences / max(self._judgements_with_sentence_stats, 1)
            if self._judgements_with_potential_sentence_stats > 0:
                agg_dict["sentence_stats_coverage"] = (
                    100.0 * self._judgements_with_sentence_stats / self._judgements_with_potential_sentence_stats
                )
            agg_dict["num_judges"] = len(self._judge_tags_seen)

        return metrics_dict
