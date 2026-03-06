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

import json
from pathlib import Path

from nemo_skills.evaluation.metrics.base import BaseMetrics


class EvalKitMetrics(BaseMetrics):
    """Metrics class for VLMEvalKit benchmarks.

    VLMEvalKit computes its own aggregate metrics during evaluation.
    This class reads pre-computed aggregates from eval_kit_metrics.json
    (written by EvalKitGenerationTask) rather than computing per-sample metrics.
    The per-sample JSONL is still read by ComputeMetrics for the update() loop,
    but we only count entries here -- the real metrics come from the JSON file.

    Note: ComputeMetrics only calls setup() on the "_all_" calculator.  When
    the data contains ``subset_for_metrics``, additional per-subset calculator
    instances are created but never receive a setup() call.  We use a
    class-level ``_shared_metrics_file`` so that those subset instances can
    still locate the eval_kit_metrics.json discovered by the "_all_" instance.
    """

    # Shared across all instances so subset calculators can find the file
    # even though only the "_all_" calculator receives setup().
    _shared_metrics_file: Path | None = None

    def __init__(self, **kwargs):
        super().__init__(compute_no_answer=False)
        self.eval_kit_metrics_file = None

    def setup(self, input_files):
        """Find the eval_kit_metrics.json in the same directory as the input files."""
        if input_files:
            # input_files are like ['/path/to/eval-results/eval_kit.MMBench_DEV_EN/output.jsonl']
            metrics_dir = Path(input_files[0]).parent
            candidate = metrics_dir / "eval_kit_metrics.json"
            if candidate.exists():
                self.eval_kit_metrics_file = candidate
                EvalKitMetrics._shared_metrics_file = candidate
            else:
                # Reset stale shared path so a previous run's file isn't reused.
                EvalKitMetrics._shared_metrics_file = None

    def update(self, predictions):
        """Count entries but don't compute per-sample metrics."""
        self.total += 1

    def get_metrics(self):
        """Return pre-computed VLMEvalKit aggregate metrics."""
        metrics_dict = {}

        # Load pre-computed metrics from VLMEvalKit.
        # Fall back to the class-level shared file for subset calculators
        # that never received a setup() call.
        eval_kit_results = {}
        effective_file = self.eval_kit_metrics_file or EvalKitMetrics._shared_metrics_file
        if effective_file and effective_file.exists():
            with open(effective_file, "rt", encoding="utf-8") as f:
                eval_kit_results = json.load(f)

        # Build the metrics in NeMo Skills format
        agg_dict = {"num_entries": self.total}

        # Flatten VLMEvalKit results into the metrics dict
        for key, value in eval_kit_results.items():
            if isinstance(value, dict):
                # Nested results (e.g., per-category scores)
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)):
                        agg_dict[f"{key}_{sub_key}"] = sub_value
            elif isinstance(value, (int, float)):
                agg_dict[key] = value

        metrics_dict["greedy"] = agg_dict
        return metrics_dict

    def metrics_to_print(self):
        return None

    def evaluations_to_print(self):
        return ["greedy"]
