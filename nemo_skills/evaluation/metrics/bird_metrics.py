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

from nemo_skills.dataset.birdbench.evaluation import sort_results, print_data
from nemo_skills.evaluation.metrics.base import BaseMetrics

class BirdMetrics(BaseMetrics):
    """Metrics for BIRD text-to-SQL evaluation."""


    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        return {"execution_accuracy": 1}


    def _compute_acc_by_diff(self, sorted_preds):
        n = len(exec_results)
        simple_results, moderate_results, challenging_results = [], [], []

        for pred in predictions:
            # Each should be a 0 or 1 value
            if pred["difficulty"] == "simple":
                simple_results.append(pred["res"])

            if pred["difficulty"] == "moderate":
                moderate_results.append(pred["res"])

            if pred["difficulty"] == "challenging":
                challenging_results.append(pred["res"])

            total_correct += pred["res"]

        simple_acc = sum(simple_results)/len(simple_results)
        moderate_acc = sum(moderate_results)/len(moderate_results)
        challenging_acc = sum(challenging_results)/len(challenging_results)

        acc = total_correct/n

        count_lists = [len(simple_results), len(moderate_results), len(challenging_results), n]
        return simple_acc * 100, moderate_acc * 100, challenging_acc * 100, acc * 100, count_lists
        

    def update(self, predictions):
        super().update(predictions)

        sorted_preds = sort_results(predictions)
        simple_acc, moderate_acc, challenging_acc, acc, count_list = \
            _compute_acc_by_diff(sorted_preds)

        print_data([simple_acc, moderate_acc, challenging_acc, acc], count_list)
        print("===========================================================================================")
        print("Finished evaluation")
