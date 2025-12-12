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

from nemo_skills.dataset.birdbench.evaluation import sort_results, compute_acc_by_diff, print_data
from nemo_skills.evaluation.metrics.base import BaseMetrics

class BirdMetrics(BaseMetrics):
    """Metrics for BIRD text-to-SQL evaluation."""

    def __init__(self, dev_json_path):
        """dev_json_path should point to the dev.json file included in the BIRD dataset download."""
        super().__init__()
        self.dev_json_path = dev_json_path

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        return {"execution_accuracy": 1}


    def update(self, predictions):
        super().update(predictions)

        sorted_preds = sort_results(predictions)
        simple_acc, moderate_acc, challenging_acc, acc, count_list = \
            compute_acc_by_diff(sorted_preds, self.dev_json_path)

        print_data([simple_acc, moderate_acc, challenging_acc, acc], count_list)
        print('===========================================================================================')
        print("Finished evaluation")
