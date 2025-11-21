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

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_float, as_int


def extract_final_cpp_block(text):
    pattern = r"```(?:cpp|Cpp)\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches[-1] if matches else ""


class ICPCMetrics(BaseMetrics):
    def __init__(self, **kwargs):
        super().__init__()
        self.reset()
        self.cluster_folder = kwargs.get("cluster_folder", None)
        print(f"Cluster folder: {self.cluster_folder}")

    def update(self, predictions):
        super().update(predictions)
        #        self._compute_pass_at_k(predictions)
        if predictions:
            self.predictions_by_problem[predictions[0]["name"]].extend(predictions)

    def _get_score_dict(self, p):
        return {"correct": all(r["score"] > 0 for r in p["test_case_results"].values())}

    def get_problem_score(self, submissions) -> bool:
        scores = []
        for submission in submissions:
            scores.append(submission["test_case_results"]["score"])
        return scores

    def get_problem_sample_score(self, submissions) -> bool:
        scores = []
        for submission in submissions:
            scores.append(submission["test_case_results"]["sample_score"])
        return scores

    def extract_info(self, submission) -> dict:
        return {
            "score": submission["test_case_results"]["score"],
            "sample_score": submission["test_case_results"]["sample_score"],
            "code": extract_final_cpp_block(submission["generation"]),
        }

    def get_clusters(self, submissions) -> dict:
        clusters = defaultdict(list)
        id = 0

        for submission in submissions:
            outputs = submission["input_case_results"]
            run_outputs = []
            for output in outputs:
                run_outputs.append(output["run_stdout"])
            output_key = tuple(run_outputs)
            extract_info = self.extract_info(submission)
            if output_key not in clusters:
                clusters[output_key] = {
                    "status": {
                        "Test passed": 0,
                        "Test failed": 0,
                        "Sample passed": 0,
                        "Sample failed": 0,
                    },
                    "codes": [],
                }
            clusters[output_key]["codes"].append(extract_info)

            id = submission["id"]
            if submission["test_case_results"]["score"] > 0:
                clusters[output_key]["status"]["Test passed"] += 1
            else:
                clusters[output_key]["status"]["Test failed"] += 1
            if submission["test_case_results"]["sample_score"] > 0:
                clusters[output_key]["status"]["Sample passed"] += 1
            else:
                clusters[output_key]["status"]["Sample failed"] += 1
        return clusters, id

    def get_metrics(self):
        self.problem_scores = {}
        self.correct_submissions = {}
        self.total_submissions = {}
        self.correct_sample_submissions = {}
        for name, submission in self.predictions_by_problem.items():
            if self.correct_submissions.get(name) is None:
                self.correct_submissions[name] = 0
            if self.total_submissions.get(name) is None:
                self.total_submissions[name] = 0
            if self.correct_sample_submissions.get(name) is None:
                self.correct_sample_submissions[name] = 0
            if self.problem_scores.get(name) is None:
                self.problem_scores[name] = False
            # Cluster the submissions
            if self.cluster_folder:
                clusters, id = self.get_clusters(submission)
                # Create the cluster_folder directory if self.cluster_folder is specified and directory does not exist
                os.makedirs(self.cluster_folder, exist_ok=True)

                # Prepare final clustered data
                final_clusters = {}

                # Convert tuple keys to string for JSON serialization
                for i, (output_key, cluster) in enumerate(clusters.items()):
                    # Compute score as sum of popularity counts for each position's output
                    final_clusters[f"cluster_{i + 1}"] = {
                        "output": output_key,
                        "status": cluster["status"],
                        "codes": cluster["codes"],
                    }

                output_file = os.path.join(self.cluster_folder, f"{id}_cluster.jsonl")
                # Write clusters to {problem_number}_cluster.jsonl
                with open(output_file, "w") as f:
                    json.dump(final_clusters, f, indent=4)

            scores = self.get_problem_score(submission)
            sample_scores = self.get_problem_sample_score(submission)
            self.correct_submissions[name] += sum(1 for value in scores if value)
            self.correct_sample_submissions[name] += sum(1 for value in sample_scores if value)

            self.total_submissions[name] += len(submission)
        metrics_dict = {}
        for name in self.total_submissions.keys():
            metrics_dict[name] = {
                "sample_correct": self.correct_sample_submissions[name],
                "test_correct": self.correct_submissions[name],
                "total": self.total_submissions[name],
            }

        metrics_dict["total"] = {
            "solved": sum(1 for value in self.correct_submissions.values() if value > 0),
            "average_number_of_runs": sum(self.total_submissions.values()) / len(self.total_submissions.values()),
        }
        return metrics_dict

    def evaluations_to_print(self):
        """Returns all problem names."""
        return ["total"] + list(self.problem_scores.keys())

    def metrics_to_print(self):
        metrics_to_print = {
            "sample_correct": as_int,
            "test_correct": as_int,
            "total": as_int,
            "solved": as_int,
            "average_number_of_runs": as_float,
        }
        return metrics_to_print

    def reset(self):
        super().reset()
        self.predictions_by_problem = defaultdict(list)
        self.problem_scores = {}

    def print_problem_scores(self):
        print("---------------------------------Problem and subtask scores---------------------------------")
        for name, scores in self.problem_scores.items():
            print(
                f"# {name}: {scores} self.correct_submissions[name]: {self.correct_submissions[name]} self.total_submissions[name]: {self.total_submissions[name]}"
            )
