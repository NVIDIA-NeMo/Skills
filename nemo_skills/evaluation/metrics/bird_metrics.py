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

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_float


class BirdMetrics(BaseMetrics):
    """Metrics for BIRD text-to-SQL evaluation."""

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        super().reset()
        self.n = 0
        self.correct = 0
        self.simple_results = []
        self.moderate_results = []
        self.challenging_results = []

    def update(self, predictions):
        self.n += len(predictions)

        for pred in predictions:
            # Each should be a 0 or 1 value
            if pred["difficulty"] == "simple":
                self.simple_results.append(pred["res"])
            elif pred["difficulty"] == "moderate":
                self.moderate_results.append(pred["res"])
            elif pred["difficulty"] == "challenging":
                self.challenging_results.append(pred["res"])

            self.correct += pred["res"]

    def get_metrics(self):
        sr = self.simple_results
        mr = self.moderate_results
        cr = self.challenging_results

        simple_acc = sum(sr) / len(sr) if sr else 0
        moderate_acc = sum(mr) / len(mr) if mr else 0
        challenging_acc = sum(cr) / len(cr) if cr else 0

        acc = self.correct / self.n if self.n else 0

        metrics_dict = {}
        metrics_dict["total"] = {
            "simple_acc": simple_acc * 100,
            "moderate_acc": moderate_acc * 100,
            "challenging_acc": challenging_acc * 100,
            "total_acc": acc * 100,
        }
        return metrics_dict

    def evaluations_to_print(self):
        return ["total"]

    def metrics_to_print(self):
        metrics_to_print = {
            "simple_acc": as_float,
            "moderate_acc": as_float,
            "challenging_acc": as_float,
            "total_acc": as_float,
        }
        return metrics_to_print
