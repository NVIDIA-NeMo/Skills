# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

from collections import defaultdict

from nemo_skills.evaluation.metrics.base import BaseMetrics


class IFMetrics(BaseMetrics):
    # loosely adapted from
    # https://github.com/google-research/google-research/blob/master/instruction_following_eval/evaluation_main.py

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        return {
            "prompt": prediction["follow_all_instructions"],
            "instruction": sum(prediction["follow_instruction_list"]),
        }

    def get_incorrect_sample(self, prediction: dict) -> dict:
        prediction = prediction.copy()
        prediction["follow_all_instructions"] = [0 for _ in prediction["follow_all_instructions"]]
        return prediction

    def update(self, predictions):
        """Updating the evaluation results with the current element.

        Args:
            predictions (list[dict]): aggregated predictions across all generations.
                The content of the file is benchmark specific.
        """
        super().update(predictions)
        self.instruction_total += len(predictions[0]["instruction_id_list"])
        strict_predictions = [pred["strict_eval"] for pred in predictions]
        loose_predictions = [pred["loose_eval"] for pred in predictions]

        self._compute_pass_at_k(predictions=strict_predictions, eval_dict=self.strict_eval_dict)
        self._compute_pass_at_k(predictions=loose_predictions, eval_dict=self.loose_eval_dict)

    def get_metrics(self):
        metrics_dict = {}
        for agg_mode in self.strict_eval_dict:
            prompt_strict = self.strict_eval_dict[agg_mode]["prompt"] / self.total * 100.0
            inst_strict = self.strict_eval_dict[agg_mode]["instruction"] / self.instruction_total * 100.0
            prompt_loose = self.loose_eval_dict[agg_mode]["prompt"] / self.total * 100.0
            inst_loose = self.loose_eval_dict[agg_mode]["instruction"] / self.instruction_total * 100.0
            metrics_dict[agg_mode] = {
                "num_prompts": self.total,
                "num_instructions": self.instruction_total,
                "average_score": (prompt_strict + inst_strict + prompt_loose + inst_loose) / 4,
                "prompt_strict_accuracy": prompt_strict,
                "instruction_strict_accuracy": inst_strict,
                "prompt_loose_accuracy": prompt_loose,
                "instruction_loose_accuracy": inst_loose,
            }
            self.update_common_metrics(metrics_dict[agg_mode])
        self._add_std_metrics(metrics_dict)
        return metrics_dict

    def reset(self):
        super().reset()
        self.instruction_total = 0
        self.strict_eval_dict = defaultdict(lambda: {"prompt": 0.0, "instruction": 0.0})
        self.loose_eval_dict = defaultdict(lambda: {"prompt": 0.0, "instruction": 0.0})
