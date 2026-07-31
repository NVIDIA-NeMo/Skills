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

import asyncio
import json
from types import SimpleNamespace

from nemo_skills.evaluation.metrics.swe_atlas_qna_metrics import (
    SweAtlasQnAMetrics,
    score_swe_atlas_qna_prediction,
)
from nemo_skills.inference.eval.swe_atlas_qna import SweAtlasQnAGenerationTask, extract_final_answer
from nemo_skills.inference.generate import GenerationTask
from nemo_skills.inference.swe_atlas_qna_judge import SweAtlasQnAJudgeTask, _extract_rating
from nemo_skills.prompt.utils import get_prompt

RUBRIC = [
    {
        "id": "positive",
        "title": "States the required behavior.",
        "annotations": {"type": "positive hli verifier", "importance": "must have"},
    },
    {
        "id": "negative",
        "title": "Claims the prohibited behavior.",
        "annotations": {"type": "negative hli verifier", "importance": "must have"},
    },
]


def _rating(criterion, status):
    return {
        "criterion_id": criterion["id"],
        "rubric_statement": criterion["title"],
        "status": status,
        "score": "1" if status == "YES" else "0",
        "justification": "Test justification.",
    }


def _prediction(positive_status="YES", negative_status="NO"):
    return {
        "rubric": json.dumps(RUBRIC),
        "judgement": json.dumps(
            {
                "ratings": [
                    _rating(RUBRIC[0], positive_status),
                    _rating(RUBRIC[1], negative_status),
                ]
            }
        ),
    }


def test_swe_atlas_qna_metrics_apply_negative_polarity():
    assert score_swe_atlas_qna_prediction(_prediction()) == {
        "task_resolved": True,
        "rubric_score": 1.0,
        "judgement_parse_error": False,
    }

    score = score_swe_atlas_qna_prediction(_prediction(negative_status="YES"))
    assert score["task_resolved"] is False
    assert score["rubric_score"] == 0.5
    assert score["judgement_parse_error"] is False


def test_swe_atlas_qna_metrics_reject_missing_or_malformed_ratings():
    missing = _prediction()
    missing["judgement"] = json.dumps({"ratings": [_rating(RUBRIC[0], "YES")]})
    malformed = _prediction()
    malformed["judgement"] = "not JSON"

    for prediction in (missing, malformed):
        score = score_swe_atlas_qna_prediction(prediction)
        assert score["task_resolved"] is False
        assert score["rubric_score"] == 0.0
        assert score["judgement_parse_error"] is True


def test_swe_atlas_qna_metrics_aggregate_task_resolve_rate():
    metrics = SweAtlasQnAMetrics()
    metrics.update([_prediction()])
    metrics.update([_prediction(negative_status="YES")])

    result = metrics.get_metrics()["pass@1"]
    assert result["task_resolved"] == 50.0
    assert result["rubric_score"] == 75.0
    assert result["judgement_parse_error"] == 0.0


def test_swe_atlas_qna_judge_rating_parser():
    criterion = RUBRIC[0]
    response = f"```json\n{json.dumps({'ratings': [_rating(criterion, 'YES')]})}\n```"
    assert _extract_rating(response, criterion) == _rating(criterion, "YES")


def test_swe_atlas_qna_judge_rating_parser_normalizes_rubric_whitespace():
    criterion = {
        **RUBRIC[0],
        "title": "States the required\nbehavior. ",
    }
    rating = _rating(criterion, "YES")
    rating["rubric_statement"] = "States the required behavior."

    parsed = _extract_rating(json.dumps({"ratings": [rating]}), criterion)

    assert parsed["rubric_statement"] == criterion["title"]


def test_swe_atlas_qna_judge_rating_parser_uses_canonical_rubric_when_wording_changes():
    criterion = {
        **RUBRIC[0],
        "title": "Reports measurements at lines 2048, 4096, and 6144.",
    }
    rating = _rating(criterion, "YES")
    rating["rubric_statement"] = "Reports measurements at lines 2048 and 6144."

    parsed = _extract_rating(json.dumps({"ratings": [rating]}), criterion)

    assert parsed["rubric_statement"] == criterion["title"]


def test_swe_atlas_qna_judge_rating_parser_repairs_trailing_commas():
    criterion = RUBRIC[0]
    response = f"""
    {{
      "ratings": [
        {{
          "criterion_id": "{criterion["id"]}",
          "rubric_statement": "{criterion["title"]}",
          "status": "YES",
          "score": "1",
          "justification": "The response demonstrates the behavior.",
        }},
      ]
    }}
    """

    assert _extract_rating(response, criterion) == _rating(criterion, "YES") | {
        "justification": "The response demonstrates the behavior."
    }


def test_swe_atlas_qna_judge_retries_empty_response(monkeypatch):
    criterion = RUBRIC[0]
    responses = iter(
        [
            {"generation": ""},
            {"generation": json.dumps({"ratings": [_rating(criterion, "YES")]})},
        ]
    )
    calls = 0

    async def fake_process(self, data_point, all_data, prompt_format=None):
        nonlocal calls
        calls += 1
        return next(responses)

    monkeypatch.setattr(GenerationTask, "process_single_datapoint", fake_process)
    task = object.__new__(SweAtlasQnAJudgeTask)
    task.cfg = SimpleNamespace(max_judgement_attempts=2, judgement_retry_delay=0)

    rating, raw_judgement, _ = asyncio.run(task._judge_criterion({}, criterion, [], None))

    assert calls == 2
    assert rating == _rating(criterion, "YES")
    assert json.loads(raw_judgement)["ratings"][0]["status"] == "YES"


def test_swe_atlas_qna_judge_records_parse_error_after_retries(monkeypatch):
    criterion = RUBRIC[0]
    calls = 0

    async def fake_process(self, data_point, all_data, prompt_format=None):
        nonlocal calls
        calls += 1
        return {"generation": ""}

    monkeypatch.setattr(GenerationTask, "process_single_datapoint", fake_process)
    task = object.__new__(SweAtlasQnAJudgeTask)
    task.cfg = SimpleNamespace(max_judgement_attempts=3, judgement_retry_delay=0)

    rating, raw_judgement, _ = asyncio.run(task._judge_criterion({}, criterion, [], None))

    assert calls == 3
    assert rating["criterion_id"] == criterion["id"]
    assert "parse_error" in rating
    assert raw_judgement == ""


def test_swe_atlas_qna_judge_calls_each_criterion(monkeypatch):
    calls = []

    async def fake_process(self, data_point, all_data, prompt_format=None):
        calls.append(data_point["criterion_id"])
        criterion = {
            "id": data_point["criterion_id"],
            "title": data_point["rubric_statement"],
        }
        return {"generation": json.dumps({"ratings": [_rating(criterion, "YES")]})}

    monkeypatch.setattr(GenerationTask, "process_single_datapoint", fake_process)
    task = object.__new__(SweAtlasQnAJudgeTask)
    output = asyncio.run(
        task.process_single_datapoint(
            {
                "problem_statement": "Question",
                "generation": "Answer",
                "rubric": json.dumps(RUBRIC),
            },
            [],
        )
    )

    judgement = json.loads(output["generation"])
    assert calls == ["positive", "negative"]
    assert [rating["criterion_id"] for rating in judgement["ratings"]] == calls


def test_swe_atlas_qna_judge_logs_expanded_criterion_prompt():
    captured = {}
    task = object.__new__(SweAtlasQnAJudgeTask)

    def fake_fill_prompt(data_point, data):
        captured.update(data_point)
        return "Rendered prompt"

    task.fill_prompt = fake_fill_prompt
    task.log_example_prompt(
        [
            {
                "problem_statement": "Question",
                "generation": "Answer",
                "rubric": json.dumps(RUBRIC),
            }
        ]
    )

    assert captured["criterion_id"] == RUBRIC[0]["id"]
    assert captured["rubric_statement"] == RUBRIC[0]["title"]
    assert captured["rubric_type"] == RUBRIC[0]["annotations"]["type"]


def test_swe_atlas_qna_judge_prompt_renders_single_criterion():
    criterion = RUBRIC[0]
    messages = get_prompt("judge/swe-atlas-qna").fill(
        {
            "problem_statement": "How does this code work?",
            "generation": "It works this way.",
            "criterion_id": criterion["id"],
            "rubric_statement": criterion["title"],
            "rubric_type": criterion["annotations"]["type"],
        }
    )
    user_message = messages[-1]["content"]
    assert "How does this code work?" in user_message
    assert "It works this way." in user_message
    assert criterion["id"] in user_message
    assert criterion["title"] in user_message
    system_message = messages[0]["content"]
    normalized_system_message = " ".join(system_message.split())
    assert "underlying technical knowledge" in system_message
    assert "missing one or two non-core items" in normalized_system_message
    assert "Do not wrap the object in Markdown fences" in normalized_system_message


def test_swe_atlas_qna_maps_agent_submission_to_generation():
    task = object.__new__(SweAtlasQnAGenerationTask)
    task.cfg = SimpleNamespace(server=SimpleNamespace(model="test-model"))
    output = task._format_mini_swe_agent_output(
        {
            "info": {
                "submission": "<<FINAL_ANSWER>>\nFinal answer\n<<FINAL_ANSWER>>",
                "exit_status": "submitted",
            }
        },
        {"instance_id": "task-1"},
    )
    assert output["generation"] == "Final answer"
    assert output["instance_id"] == "task-1"
    assert output["model_name_or_path"] == "test-model"


def test_swe_atlas_qna_final_answer_extraction_falls_back_to_plain_submission():
    assert extract_final_answer("  Plain final answer  ") == "Plain final answer"
    assert extract_final_answer(None) == ""
