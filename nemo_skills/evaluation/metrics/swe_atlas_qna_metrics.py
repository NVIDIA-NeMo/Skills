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

import json

from nemo_skills.evaluation.metrics.base import BaseMetrics, as_int, as_percentage


def _load_json_field(value, field_name):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError(f"{field_name} is not valid JSON") from error
    return value


def score_swe_atlas_qna_prediction(prediction: dict) -> dict[str, bool | float]:
    """Score one judged prediction using SWE-Atlas-QnA rubric polarity."""
    try:
        rubric = _load_json_field(prediction["rubric"], "rubric")
        judgement = _load_json_field(prediction["judgement"], "judgement")
        ratings = judgement["ratings"]
        if not isinstance(rubric, list) or not rubric or not isinstance(ratings, list):
            raise ValueError("rubric and judgement.ratings must be non-empty lists")

        ratings_by_id = {}
        for rating in ratings:
            criterion_id = rating["criterion_id"]
            if criterion_id in ratings_by_id:
                raise ValueError(f"duplicate rating for criterion {criterion_id}")
            ratings_by_id[criterion_id] = rating

        rubric_ids = {criterion["id"] for criterion in rubric}
        if set(ratings_by_id) != rubric_ids:
            raise ValueError("judge ratings do not match all rubric criterion IDs")

        passed = 0
        for criterion in rubric:
            rating = ratings_by_id[criterion["id"]]
            if rating.get("parse_error"):
                raise ValueError(f"unparseable rating for criterion {criterion['id']}")
            if rating.get("rubric_statement") != criterion["title"]:
                raise ValueError(f"rubric statement mismatch for criterion {criterion['id']}")

            status = rating.get("status")
            score = str(rating.get("score"))
            if status not in ("YES", "NO") or score not in ("0", "1"):
                raise ValueError(f"invalid rating for criterion {criterion['id']}")
            if (status == "YES") != (score == "1"):
                raise ValueError(f"inconsistent status and score for criterion {criterion['id']}")

            behavior_present = score == "1"
            rubric_type = criterion.get("annotations", {}).get("type", "")
            criterion_passed = not behavior_present if "negative" in rubric_type.lower() else behavior_present
            passed += int(criterion_passed)

        rubric_score = passed / len(rubric)
        return {
            "task_resolved": rubric_score == 1.0,
            "rubric_score": rubric_score,
            "judgement_parse_error": False,
        }
    except (KeyError, TypeError, ValueError):
        return {
            "task_resolved": False,
            "rubric_score": 0.0,
            "judgement_parse_error": True,
        }


class SweAtlasQnAMetrics(BaseMetrics):
    def __init__(self):
        super().__init__(compute_no_answer=False)

    def _get_score_dict(self, prediction: dict) -> dict[str, bool | float]:
        return score_swe_atlas_qna_prediction(prediction)

    def get_incorrect_sample(self, prediction: dict) -> dict:
        prediction = prediction.copy()
        prediction["judgement"] = ""
        return prediction

    def update(self, predictions):
        super().update(predictions)
        self._compute_pass_at_k(predictions=predictions)

    def evaluations_to_print(self):
        return [f"pass@1[avg-of-{self.max_k}]", f"pass@{self.max_k}"]

    def metrics_to_print(self):
        return {
            "num_entries": as_int,
            "avg_tokens": as_int,
            "gen_seconds": as_int,
            "task_resolved": as_percentage,
            "rubric_score": as_percentage,
            "judgement_parse_error": as_percentage,
        }
