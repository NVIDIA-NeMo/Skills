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

"""Aggregation of SHERLOC fault-localization scores into reportable metrics.

This module turns the per-instance ``eval_status`` blocks written by the SHERLOC evaluator
(``recipes.sherloc.evaluation.sherloc``) into the aggregate numbers reported for a run.
It is selected through ``METRICS_TYPE = "sherloc"`` in the benchmark definition.

Mapping to the paper's tables
-----------------------------
Every per-generation score lives in ``eval_status[0]``, under ``file_level`` (file-set
comparison) and ``chunk_containment`` (line-range comparison). This class averages each score
unweighted over the *scored generations*: those whose gold patch yields at least one location.
Generations with no gold locations are counted separately under ``samples_no_gt`` and are left
out of every average, while instances the pipeline failed or skipped keep their all-zero score
block and so do count against the averages. With a single generation per instance, which is
the setting the paper reports, one scored generation is one benchmark instance. The reported
keys are then scaled to percentages:

* ``file_recall`` - mean over scored instances of
  ``|gold_files ∩ predicted_files| / |gold_files|``.
  This is the paper's **Recall@1** on SWE-Bench Verified. On SWE-Bench Lite, whose gold patches
  touch exactly one file, the per-instance value is 0 or 1 and the mean is the paper's
  **Accuracy@1**: the percentage of instances in which the single gold file was predicted.
* ``file_precision`` - mean of ``|gold_files ∩ predicted_files| / |predicted_files|``.
* ``file_f1`` - mean of the per-instance F1 of the two above. Note this is a mean of
  per-instance F1 values, not the F1 of the mean precision and recall.
* ``file_accuracy`` - mean per-instance Jaccard index of the two file sets. Despite the name
  this is *not* the paper's Accuracy@1; use ``file_recall`` for that.
* ``file_exact_match`` - fraction of instances whose predicted file set equals the gold set
  exactly.
* ``chunk_coverage_recall`` - mean fraction of gold hunks fully enclosed by some predicted
  line range in the same file.
* ``chunk_precision`` - mean fraction of predicted ranges that enclose at least one gold hunk.
* ``chunk_avg_tightness`` - mean ratio of gold-hunk length to the length of the tightest
  enclosing prediction, averaged over enclosed hunks only. 100% means predictions matched the
  gold ranges exactly; lower values mean they were padded with surrounding context.

The ``@1`` in the paper's metric names refers to a single generation per instance, i.e. the
``pass@1`` aggregation mode.
"""

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_percentage


class SherlocMetrics(BaseMetrics):
    """Aggregates per-instance SHERLOC localization scores over a benchmark run."""

    def __init__(self):
        super().__init__()
        self.file_level_metrics = []
        self.chunk_containment_metrics = []
        self.total_gt_locations = 0
        self.total_pred_locations = 0
        self.successful_samples = 0
        self.failed_samples = 0
        self.no_gt_samples = 0  # Instances whose gold patch yielded no locations.

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        """Extract the per-generation scores that feed the pass@k machinery.

        Returns the file-level and chunk-level scores of a single generation, defaulting to
        zero when ``eval_status`` is missing or malformed so that an unparsable generation is
        scored as a complete miss rather than dropped.
        """
        eval_status = prediction.get("eval_status", [])

        if not eval_status or not isinstance(eval_status[0], dict):
            # Fallback for an empty or malformed eval_status.
            return {
                "file_precision": 0.0,
                "file_recall": 0.0,
                "file_f1": 0.0,
                "file_exact_match": 0.0,
                "file_accuracy": 0.0,
                "chunk_coverage_recall": 0.0,
                "chunk_avg_tightness": 0.0,
                "chunk_precision": 0.0,
            }

        eval_result = eval_status[0]
        file_metrics = eval_result.get("file_level", {})
        chunk_metrics = eval_result.get("chunk_containment", {})

        return {
            "file_precision": file_metrics.get("precision", 0.0),
            "file_recall": file_metrics.get("recall", 0.0),
            "file_f1": file_metrics.get("f1", 0.0),
            "file_exact_match": file_metrics.get("exact_match", 0.0),
            "file_accuracy": file_metrics.get("accuracy", 0.0),
            "chunk_coverage_recall": chunk_metrics.get("coverage_recall", 0.0),
            "chunk_avg_tightness": chunk_metrics.get("avg_prediction_tightness", 0.0),
            "chunk_precision": chunk_metrics.get("precision", 0.0),
        }

    @classmethod
    def get_incorrect_sample(cls, prediction: dict) -> dict:
        """Return an ``eval_status`` block that scores as a complete miss.

        Used by length-based filtering to grade an over-long generation as incorrect without
        re-running the evaluator on it.
        """
        return {
            "eval_status": [
                {
                    "file_level": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "exact_match": 0.0, "accuracy": 0.0},
                    "chunk_containment": {
                        "coverage_recall": 0.0,
                        "avg_prediction_tightness": 0.0,
                        "precision": 0.0,
                        "covered_chunks": 0,
                        "total_chunks": 0,
                        "useful_predictions": 0,
                        "total_predictions": 0,
                    },
                    "ground_truth_count": 0,
                    "predicted_count": 0,
                    "is_failed_sample": True,
                }
            ]
        }

    def update(self, predictions):
        """Accumulate the scores of one instance, given as its list of generations."""
        # The base class tracks entry counts and token statistics.
        super().update(predictions)

        # Adopt the evaluator's own success/failure counts, which are attached to the first
        # entry of the generation file. Only read them once, so that a second batch does not
        # overwrite the totals already collected.
        if (
            predictions
            and "processing_stats" in predictions[0]
            and self.successful_samples == 0
            and self.failed_samples == 0
        ):
            processing_stats = predictions[0]["processing_stats"]
            self.successful_samples = processing_stats.get("successful_samples", 0)
            self.failed_samples = processing_stats.get("failed_samples", 0)

        # Collect the per-instance scores that the aggregate averages are computed over.
        for prediction in predictions:
            eval_status = prediction.get("eval_status", [])
            if eval_status and isinstance(eval_status[0], dict):
                eval_result = eval_status[0]

                # Instances with no gold locations have no meaningful recall, so they are
                # counted separately and left out of every average.
                gt_count = eval_result.get("ground_truth_count", 0)
                if gt_count == 0:
                    self.no_gt_samples += 1
                    continue

                # Fall back to counting successes and failures here when processing_stats was
                # absent or covered fewer entries than have been seen.
                if self.successful_samples + self.failed_samples < self.total:
                    # Prefer the explicit marker written by the evaluator; otherwise infer.
                    is_failed_sample = eval_result.get("is_failed_sample", False)
                    if not is_failed_sample:
                        # The generation status, when present, is the next best signal.
                        if "status" in prediction and prediction["status"] != "success":
                            is_failed_sample = True
                        else:
                            # Last resort: a generation that predicted nothing and scored zero.
                            predicted_count = eval_result.get("predicted_count", 0)
                            file_metrics = eval_result.get("file_level", {})
                            is_failed_sample = (
                                predicted_count == 0
                                and file_metrics.get("precision", 0) == 0
                                and file_metrics.get("recall", 0) == 0
                                and file_metrics.get("f1", 0) == 0
                            )

                    if is_failed_sample:
                        self.failed_samples += 1
                    else:
                        self.successful_samples += 1

                # Keep the raw score dictionaries; they are averaged in get_metrics.
                if "file_level" in eval_result:
                    self.file_level_metrics.append(eval_result["file_level"])
                if "chunk_containment" in eval_result:
                    self.chunk_containment_metrics.append(eval_result["chunk_containment"])

                self.total_gt_locations += eval_result.get("ground_truth_count", 0)
                self.total_pred_locations += eval_result.get("predicted_count", 0)

        self._compute_pass_at_k(predictions=predictions)

    def _compute_aggregate_metrics(self, metrics_list):
        """Average a list of per-instance score dictionaries entry by entry.

        The set of keys to average is inferred from the first dictionary, so the same helper
        serves the file-level, chunk-containment and line-level score shapes. Each average is
        taken over the entries that actually carry the key, and the result stays in ``[0, 1]``
        (``get_metrics`` scales it to a percentage).
        """
        if not metrics_list:
            return {}

        aggregate = {}

        # File-level score keys.
        file_metric_names = ["precision", "recall", "f1", "exact_match", "accuracy"]

        # Chunk-containment score keys.
        chunk_metric_names = ["coverage_recall", "avg_prediction_tightness", "precision"]

        # Infer which score shape this list holds from a marker key of the first entry.
        if metrics_list and len(metrics_list) > 0:
            first_item = metrics_list[0]
            if "accuracy" in first_item:
                # File-level scores.
                for metric in file_metric_names:
                    values = [m.get(metric, 0.0) for m in metrics_list if metric in m]
                    aggregate[metric] = sum(values) / len(values) if values else 0.0
            elif "coverage_recall" in first_item:
                # Chunk-containment scores.
                for metric in chunk_metric_names:
                    values = [m.get(metric, 0.0) for m in metrics_list if metric in m]
                    aggregate[metric] = sum(values) / len(values) if values else 0.0
            else:
                # Any other score shape.
                for metric in ["precision", "recall", "f1", "exact_match"]:
                    values = [m.get(metric, 0.0) for m in metrics_list if metric in m]
                    aggregate[metric] = sum(values) / len(values) if values else 0.0
        return aggregate

    def reset(self):
        """Clear all accumulated scores and counters."""
        super().reset()
        self.file_level_metrics = []
        self.chunk_containment_metrics = []
        self.total_gt_locations = 0
        self.total_pred_locations = 0
        self.successful_samples = 0
        self.failed_samples = 0
        self.no_gt_samples = 0

    def metrics_to_print(self):
        """Select and format the columns of the summary table."""

        def format_count(key, value, all_metrics):
            return str(int(value))

        return {
            "num_entries": format_count,
            "samples_with_gt": format_count,
            "samples_no_gt": format_count,
            "total_gt_locs": format_count,
            "total_pred_locs": format_count,
            "file_precision": as_percentage,
            "file_recall": as_percentage,
            "file_f1": as_percentage,
            "file_accuracy": as_percentage,
            "chunk_coverage_recall": as_percentage,
            "chunk_avg_tightness": as_percentage,
            "chunk_precision": as_percentage,
        }

    def get_metrics(self):
        """Return the aggregate metrics, keyed by aggregation mode.

        The base class contributes the pass@k and majority@k views of the scores returned by
        ``_get_score_dict``. On top of those, the run-level counters and the unweighted
        averages described in the module docstring are attached to every aggregation mode,
        already scaled to percentages. The averages are computed over instances that have at
        least one gold location (``samples_with_gt``), which is the denominator behind the
        paper's Accuracy@1 and Recall@1.
        """
        metrics_dict = super().get_metrics()

        aggregate_file = self._compute_aggregate_metrics(self.file_level_metrics)
        aggregate_chunk = self._compute_aggregate_metrics(self.chunk_containment_metrics)

        for agg_mode in metrics_dict.keys():
            # self.total is the number of entries the metrics object has seen.
            total_samples = self.total
            # Instances with no gold locations are excluded from every average.
            effective_total = total_samples - self.no_gt_samples

            # Prefer explicit success/failure counts when present.
            if self.successful_samples > 0 or self.failed_samples > 0:
                counted_samples = self.successful_samples + self.failed_samples
                # If the counts cover only part of the run, report the rate over that part.
                if counted_samples < effective_total:
                    success_rate = (self.successful_samples / counted_samples * 100) if counted_samples > 0 else 0.0
                else:
                    success_rate = (self.successful_samples / effective_total * 100) if effective_total > 0 else 0.0
            else:
                # Without explicit counts, treat every scored instance as successful.
                success_rate = 100.0 if effective_total > 0 else 0.0

            metrics_dict[agg_mode].update(
                {
                    # Processing statistics.
                    "total_samples": total_samples,
                    "samples_with_gt": effective_total,
                    "samples_no_gt": self.no_gt_samples,
                    "successful_samples": self.successful_samples,
                    "failed_samples": self.failed_samples,
                    "success_rate": success_rate,
                    # Summary statistics, under the shorter names used by the table.
                    "total_gt_locs": self.total_gt_locations,
                    "total_pred_locs": self.total_pred_locations,
                    "avg_gt_per_case": self.total_gt_locations / effective_total if effective_total > 0 else 0.0,
                    "avg_pred_per_case": self.total_pred_locations / effective_total if effective_total > 0 else 0.0,
                    # File-level averages, in percent.
                    "file_precision": aggregate_file.get("precision", 0.0) * 100,
                    "file_recall": aggregate_file.get("recall", 0.0) * 100,
                    "file_f1": aggregate_file.get("f1", 0.0) * 100,
                    "file_exact_match": aggregate_file.get("exact_match", 0.0) * 100,
                    "file_accuracy": aggregate_file.get("accuracy", 0.0) * 100,
                    # Chunk-containment averages, in percent.
                    "chunk_coverage_recall": aggregate_chunk.get("coverage_recall", 0.0) * 100,
                    "chunk_avg_tightness": aggregate_chunk.get("avg_prediction_tightness", 0.0) * 100,
                    "chunk_precision": aggregate_chunk.get("precision", 0.0) * 100,
                }
            )

        return metrics_dict
