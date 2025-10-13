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


class IOIMetrics(BaseMetrics):
    def __init__(self):
        super().__init__()
        self.reset()

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions)
        if predictions:
            self.predictions_by_problem[predictions[0]["name"]].extend(predictions)

    def _get_score_dict(self, p):
        return {"correct": all(r["score"] > 0 for r in p["test_case_results"].values())}

    def get_problem_score(self, submissions) -> float:
        """
        For a given problem (list of submissions), compute the score as follows:
          - For each subtask, take the maximum score over all submissions.
          - Sum these maximum scores to get the problem score.
        """
        if not submissions:
            return 0.0
        subtask_scores = {}

        for submission in submissions:
            for subtask, result in submission["test_case_results"].items():
                subtask_scores[subtask] = max(subtask_scores.get(subtask, 0), result["score"])
        return sum(subtask_scores.values()), subtask_scores

    def simulate_round_robin_score(self, submissions) -> float:
        """
        Computes a round robin score for a problem.
        The procedure is as follows:
         1. For each submission, compute an aggregate score (sum of subtask scores).
         2. Sort submissions in descending order by the aggregate score.
         3. Select up to 50 submissions.
         4. For each subtask, take the maximum score among the selected submissions.
         5. Return the sum of these maximum subtask scores.
        """
        if not submissions:
            return 0.0

        # compute an aggregate score per submission
        for submission in submissions:
            aggregate_score = sum(result["score"] for result in submission["test_case_results"].values())
            submission["_aggregate_score"] = aggregate_score

        # sort submissions in descending order by aggregate score
        sorted_submissions = sorted(submissions, key=lambda s: s["_aggregate_score"], reverse=True)
        # Select up to 50 submissions.
        selected = sorted_submissions[:50]

        # for each subtask, take the maximum score among the selected submissions
        subtask_scores = {}
        for submission in selected:
            for subtask, result in submission["test_case_results"].items():
                subtask_scores[subtask] = max(subtask_scores.get(subtask, 0), result["score"])
        return sum(subtask_scores.values())

    def get_metrics(self):
        total_score = total_round_robin = 0.0
        self.problem_scores = {}
        for name, submissions in self.predictions_by_problem.items():
            score, subtasks = self.get_problem_score(submissions)
            self.problem_scores[name] = (score, subtasks)
            total_score += score
            total_round_robin += self.simulate_round_robin_score(submissions)
        self.print_problem_scores()
        metrics_dict = super().get_metrics()
        for m in metrics_dict.values():
            m["total_score"], m["round_robin_score"] = str(total_score), str(total_round_robin)
        return metrics_dict

    def reset(self):
        super().reset()
        self.predictions_by_problem = defaultdict(list)
        self.problem_scores = {}

    def print_problem_scores(self):
        print("---------------------------------Problem and subtask scores---------------------------------")
        for name, (achieved_total, achieved_subtasks) in self.problem_scores.items():
            submissions = self.predictions_by_problem[name]
            max_subtasks = {}
            for sub in submissions:
                max_subtasks[sub["subtask"]] = sub["subtask_score"]
            max_total = sum(max_subtasks.values())
            print(f"# {name}: {achieved_total}/{max_total}")
            for subtask, achieved in achieved_subtasks.items():
                print(f"  {subtask}: {achieved}/{max_subtasks[subtask]}")
