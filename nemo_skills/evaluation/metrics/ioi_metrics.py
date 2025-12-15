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
import json
import os
import re
from collections import defaultdict

from nemo_skills.evaluation.metrics.base import BaseMetrics


def extract_final_cpp_block(text):
    pattern = r"```(?:cpp|Cpp)\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1] if matches else ""


class IOIMetrics(BaseMetrics):
    def __init__(self, **kwargs):
        super().__init__()
        self.reset()
        self.cluster_folder = kwargs.get("cluster_folder", None)
        print(f"Cluster folder: {self.cluster_folder}")

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions)
        if predictions:
            self.predictions_by_problem[predictions[0]["name"]].extend(predictions)

    def _get_score_dict(self, p):
        return {"correct": all(r["score"] > 0 for r in p["test_case_results"].values())}

    def extract_info(self, submission) -> dict:
        # Aggregate IOI per-submission scores for convenience
        subtask_scores = [v["score"] for _, v in submission["test_case_results"].items()]
        return {
            "grade": subtask_scores,
            "tokens": submission["num_generated_tokens"],
            "code": extract_final_cpp_block(submission["generation"]),
        }

    def get_clusters(self, submissions) -> dict:
        clusters = defaultdict(list)
        id = 0

        for submission in submissions:
            input_results = submission.get("input_case_results", [])
            run_outputs = []
            for output in input_results:
                if "run_stdout" not in output:
                    continue
                run_outputs.append(output["run_stdout"])
            output_key = tuple(run_outputs)

            extract_info = self.extract_info(submission)
            if output_key not in clusters:
                # Initialize per-subtask maxima and counts with this submission's scores
                subtask_score_list = [res["score"] for _, res in submission["test_case_results"].items()]
                clusters[output_key] = {
                    "codes": [],
                    "max_score": subtask_score_list[:],
                    "max_score_solutions": [1] * len(subtask_score_list),
                }
            else:
                # Update maxima and counts element-wise from this submission
                subtask_score_list = [res["score"] for _, res in submission["test_case_results"].items()]
                max_scores = clusters[output_key]["max_score"]
                max_counts = clusters[output_key]["max_score_solutions"]
                for idx, score_val in enumerate(subtask_score_list):
                    if score_val > max_scores[idx]:
                        max_scores[idx] = score_val
                        max_counts[idx] = 1
                    elif score_val == max_scores[idx]:
                        max_counts[idx] += 1
            clusters[output_key]["codes"].append(extract_info)

            id = submission.get("id", id)

        return clusters, id

    def get_problem_score(self, submissions) -> float:
        """
        For a given problem (list of submissions), compute the score as follows:
          - For each subtask, take the maximum score over all submissions.
          - Sum these maximum scores to get the problem score.
        """
        if not submissions:
            return 0.0, {}
        subtask_scores = {}

        for submission in submissions:
            for subtask, result in submission["test_case_results"].items():
                subtask_scores[subtask] = max(subtask_scores.get(subtask, 0), result["score"])
        return sum(subtask_scores.values()), subtask_scores

    def get_metrics(self):
        total_score = 0.0
        self.problem_scores = {}
        for name, submissions in self.predictions_by_problem.items():
            # Cluster the submissions if requested
            if self.cluster_folder:
                os.makedirs(self.cluster_folder, exist_ok=True)
                submissions_by_id = defaultdict(list)
                for sub in submissions:
                    submissions_by_id[sub["id"]].append(sub)
                for sid, sid_submissions in submissions_by_id.items():
                    clusters, _ = self.get_clusters(sid_submissions)
                    final_clusters = {}
                    for i, (output_key, cluster) in enumerate(clusters.items()):
                        final_clusters[f"cluster_{i + 1}"] = {
                            "output": output_key,
                            "codes": cluster["codes"],
                            "max_score": cluster["max_score"],
                            "max_score_solutions": cluster["max_score_solutions"],
                        }
                    output_file = os.path.join(self.cluster_folder, f"{sid}_cluster.jsonl")
                    with open(output_file, "w") as f:
                        json.dump(final_clusters, f, indent=4)

            score, subtasks = self.get_problem_score(submissions)
            self.problem_scores[name] = (score, subtasks)
            total_score += score

        per_problem_subtask_scores = {}
        for name, (achieved_total, achieved_subtasks) in self.problem_scores.items():
            submissions = self.predictions_by_problem[name]
            max_subtasks = {}
            for sub in submissions:
                max_subtasks[sub["subtask"]] = sub["subtask_score"]
            max_total = sum(max_subtasks.values())
            per_problem_subtask_scores[name] = {
                "total": {"score": achieved_total, "max_score": max_total},
                "subtasks": {
                    subtask: {"score": achieved, "max_score": max_subtasks[subtask]}
                    for subtask, achieved in achieved_subtasks.items()
                },
            }

        metrics_dict = super().get_metrics()
        for m in metrics_dict.values():
            m["total_score"] = int(total_score)
            m["per_problem_subtask_scores"] = per_problem_subtask_scores
        self.per_problem_subtask_scores = per_problem_subtask_scores
        self.print_problem_scores()
        return metrics_dict

    def reset(self):
        super().reset()
        self.predictions_by_problem = defaultdict(list)
        self.problem_scores = {}
        self.per_problem_subtask_scores = {}

    def evaluations_to_print(self):
        return [f"pass@{self.max_k}"]

    def print_problem_scores(self):
        print("---------------------------------Problem and subtask scores---------------------------------")
        for name, info in self.per_problem_subtask_scores.items():
            total = info["total"]
            print(f"# {name}: {int(total['score'])}/{int(total['max_score'])}")
            for subtask, subinfo in info["subtasks"].items():
                print(f"  {subtask}: {int(subinfo['score'])}/{int(subinfo['max_score'])}")
