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

from nemo_skills.evaluation.metrics.base import BaseMetrics


class HilSweMetrics(BaseMetrics):
    """Metrics for HIL-Bench (SWE domain).

    Two complementary signals:

    * pass@k - whether the task was resolved (computed per-instance over k passes using the
      standard BaseMetrics pass@k machinery, scored on ``swe-bench-metrics.resolved``).
    * Ask-F1 - the harmonic mean of question precision and blocker recall. Following the public
      HIL-Bench repo (NOT the paper-text formula), questions are deduplicated by matched
      ``blocker_name`` into a ``discovered`` count, and that same count is the numerator for
      BOTH precision (discovered / all questions) and recall (discovered / blockers present).
      Using ``discovered`` rather than the raw count of relevant questions matters when several
      relevant questions hit the same blocker -- the repo counts that blocker once. This is a
      GLOBAL, pooled process metric: totals are accumulated across all instances and passes,
      then a single precision/recall/F1 is reported. It is reported alongside every printed
      pass@k block.
    """

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        resolved = prediction.get("swe-bench-metrics", {}).get("resolved")
        return {"issues_resolved": bool(resolved)}

    def get_incorrect_sample(self, prediction: dict) -> dict:
        return {
            "swe-bench-metrics": {
                "resolved": False,
                "patch_exists": True,
                "patch_successfully_applied": True,
            },
            "ask_human_log": {"questions": [], "n_blockers": 0, "blockers": {}},
        }

    def reset(self):
        super().reset()
        # Pooled ask_human totals (global precision/recall/F1).
        self.ask_questions_total = 0
        self.ask_relevant_questions_total = 0
        self.ask_discovered_total = 0
        self.ask_blockers_total = 0
        # Count of per-(instance,seed) predictions classified as infra failures by the generation
        # module (status="infra_error"). The orchestrator's resume/rerun loop should drive this to
        # ~0; a non-zero residual means some passes survived all rerun attempts and -- like
        # upstream's never-fixed passes -- should be treated as excluded, not as capability misses.
        self.infra_error_total = 0

    def _accumulate_ask(self, predictions):
        for pred in predictions:
            log = pred.get("ask_human_log") or {}
            questions = log.get("questions") or log.get("entries") or []
            relevant_questions = [q for q in questions if q.get("blocker_name") is not None]
            discovered = {q["blocker_name"] for q in relevant_questions}
            self.ask_questions_total += len(questions)
            self.ask_relevant_questions_total += len(relevant_questions)
            self.ask_discovered_total += len(discovered)
            self.ask_blockers_total += int(log.get("n_blockers", 0))

    def _accumulate_infra(self, predictions):
        for pred in predictions:
            is_infra = bool(pred.get("infra_error")) or pred.get("status") == "infra_error" or (
                (pred.get("swe-bench-metrics") or {}).get("resolved") is None
                and (pred.get("swe-bench-metrics") or {}).get("patch_successfully_applied") is None
            )
            if is_infra:
                self.infra_error_total += 1

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions=predictions)
        self._accumulate_ask(predictions)
        self._accumulate_infra(predictions)

    def get_metrics(self):
        metrics_dict = super().get_metrics()

        q = self.ask_questions_total
        r = self.ask_relevant_questions_total
        d = self.ask_discovered_total
        b = self.ask_blockers_total
        # Repo behavior: the deduplicated ``discovered`` count is the numerator for BOTH
        # precision and recall (not ``relevant_questions / all_questions`` as in the paper text).
        precision = d / q if q > 0 else 0.0
        recall = d / b if b > 0 else 0.0
        ask_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        ask_metrics = {
            # Float values are already scaled to percentages here (get_metrics does not
            # rescale entries we add after super()).
            "ask_f1": ask_f1 * 100.0,
            "ask_precision": precision * 100.0,
            "ask_recall": recall * 100.0,
            "num_questions": q,
            "num_relevant_questions": r,
            "num_blockers": b,
            "num_blockers_discovered": d,
            "num_infra_error": self.infra_error_total,
        }
        # Surface the pooled Ask-F1 alongside every aggregation block that gets printed.
        for agg_mode in metrics_dict:
            metrics_dict[agg_mode].update(ask_metrics)
        return metrics_dict

    def metrics_to_print(self):
        return None
