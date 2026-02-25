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

from nemo_skills.evaluation.metrics.base import BaseMetrics


class EvalPlusMetrics(BaseMetrics):
    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        return {
            "passing_base_tests": prediction["is_correct"],
            "passing_plus_tests": prediction["is_correct-plus"],
        }

    def get_incorrect_sample(self, prediction: dict) -> dict:
        return {"is_correct": False, "is_correct-plus": False}

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions=predictions)


class LiveCodeBenchMetrics(BaseMetrics):
    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        return {
            "accuracy": prediction["graded_list"][0],
        }

    def get_incorrect_sample(self, prediction: dict) -> dict:
        return {"graded_list": [False]}

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions=predictions)


class SweBenchMetrics(BaseMetrics):
    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        return {
            "issues_resolved": prediction["swe-bench-metrics"]["resolved"],
            "no_patch": not prediction["swe-bench-metrics"]["patch_exists"],
            "patch_cant_apply": not prediction["swe-bench-metrics"]["patch_successfully_applied"],
        }

    def get_incorrect_sample(self, prediction: dict) -> dict:
        return {"swe-bench-metrics": {"resolved": False, "patch_exists": True, "patch_successfully_applied": True}}

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions=predictions)


class SciCodeMetrics(BaseMetrics):
    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        subtask_status_list = prediction["eval_status"]
        correct_subtasks = sum(subtask["process_status"] == "completed" for subtask in subtask_status_list)
        return {
            "problem_accuracy": correct_subtasks == len(subtask_status_list),
            "subtask_accuracy": correct_subtasks,
        }

    def get_incorrect_sample(self, prediction: dict) -> dict:
        prediction = prediction.copy()
        subtask_status_list = prediction["eval_status"]
        for subtask in subtask_status_list:
            subtask["process_status"] = "error"
        prediction["eval_status"] = subtask_status_list
        return prediction

    def update(self, predictions):
        super().update(predictions)
        self.subtasks_total += len(predictions[0]["eval_status"])
        self._compute_pass_at_k(predictions)

    def get_metrics(self):
        metrics_dict = super().get_metrics()
        for agg_mode in self.eval_dict.keys():
            metrics_dict[agg_mode]["num_problems"] = metrics_dict[agg_mode].pop("num_entries")
            metrics_dict[agg_mode]["num_subtasks"] = self.subtasks_total
            # correcting subtask normalization
            metrics_dict[agg_mode]["subtask_accuracy"] *= self.total / self.subtasks_total

        return metrics_dict

    def reset(self):
        super().reset()
        self.subtasks_total = 0


class BigCodeBenchMetrics(BaseMetrics):
    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        return {
            "accuracy": prediction["status"] == "pass",
        }

    def get_incorrect_sample(self, prediction: dict) -> dict:
        return {"status": "fail"}

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions=predictions)


class HumanEvalInfillingMetrics(BaseMetrics):
    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        return {"accuracy": prediction["passed"]}

    def get_incorrect_sample(self, prediction: dict) -> dict:
        return {"passed": False}

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions=predictions)


class ComputeEvalMetrics(BaseMetrics):
    def _get_score_dict(self, prediction: dict) -> dict[str, bool | int | float]:
        return {"accuracy": prediction["passed"]}

    def get_incorrect_sample(self, prediction: dict) -> dict:
        return {"passed": False}

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions=predictions)
