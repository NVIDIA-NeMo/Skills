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

import pytest
import yaml

from nemo_skills.evaluation.metrics.swe_atlas_qna_metrics import (
    SweAtlasQnAMetrics,
    score_swe_atlas_qna_prediction,
)
from nemo_skills.inference.eval.opencode_utils import (
    build_opencode_config,
    build_opencode_install_command,
    extract_final_assistant_text,
    extract_final_assistant_text_from_jsonl,
)
from nemo_skills.inference.eval.swe_atlas_qna import SweAtlasQnAGenerationTask, extract_final_answer
from nemo_skills.inference.eval.swebench import (
    SupportedAgentFrameworks,
    SweBenchGenerationTask,
    _override_mini_swe_agent_cwd,
)
from nemo_skills.inference.generate import GenerationTask, GenerationTaskConfig
from nemo_skills.inference.swe_atlas_qna_judge import SweAtlasQnAJudgeTask, _extract_rating
from nemo_skills.prompt.utils import get_config_path, get_prompt

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


def test_mini_swe_agent_cwd_override_preserves_yaml_default():
    config = {"environment": {"cwd": "/app"}}

    _override_mini_swe_agent_cwd(config, None)
    assert config["environment"]["cwd"] == "/app"

    _override_mini_swe_agent_cwd(config, "/testbed")
    assert config["environment"]["cwd"] == "/testbed"

    config_without_environment = {}
    _override_mini_swe_agent_cwd(config_without_environment, "/testbed")
    assert config_without_environment["environment"]["cwd"] == "/testbed"


def test_continue_on_error_defaults_to_false():
    cfg = GenerationTaskConfig(
        input_file="input.jsonl",
        output_file="output.jsonl",
        prompt_format="openai",
        server={"server_type": "openai"},
    )

    assert cfg.continue_on_error is False


def test_continue_on_error_writes_successful_subset_and_error_sidecar(tmp_path):
    async def run():
        task = object.__new__(GenerationTask)
        task.cfg = SimpleNamespace(
            output_file=str(tmp_path / "output.jsonl"),
            continue_on_error=True,
            async_position_key="_async_position",
            add_generation_stats=False,
            generation_key="generation",
            parse_reasoning=False,
            end_reasoning_string="</think>",
            drop_content_types=[],
            enable_litellm_cache=False,
        )
        task.output_lock = None
        task.should_run_evaluation = False
        task.evaluator = None
        task._reasoning_warning_shown = False

        async def process_single_datapoint(data_point, all_data, prompt_format=None):
            if data_point["instance_id"] == "failed":
                raise RuntimeError("context length exceeded")
            return {"generation": f"answer-{data_point['instance_id']}"}

        task.process_single_datapoint = process_single_datapoint
        data = [
            {"instance_id": "first", "_async_position": 0},
            {"instance_id": "failed", "_async_position": 1},
            {"instance_id": "last", "_async_position": 2},
        ]
        await task.async_loop(data)

    asyncio.run(run())

    output_rows = [json.loads(line) for line in (tmp_path / "output.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [row["instance_id"] for row in output_rows] == ["first", "last"]
    assert [row["generation"] for row in output_rows] == ["answer-first", "answer-last"]

    error_rows = [
        json.loads(line) for line in (tmp_path / "output.errors.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert error_rows[0]["instance_id"] == "failed"
    assert error_rows[0]["error_type"] == "RuntimeError"
    assert error_rows[0]["error_message"] == "context length exceeded"
    assert "RuntimeError: context length exceeded" in error_rows[0]["traceback"]


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
    task.cfg = SimpleNamespace(max_judgement_attempts=1, judgement_retry_delay=0)
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


def test_swe_atlas_qna_maps_swe_agent_submission_to_generation():
    task = object.__new__(SweAtlasQnAGenerationTask)
    task.cfg = SimpleNamespace(server=SimpleNamespace(model="test-model"))
    output = task._format_swe_agent_output(
        {
            "model_patch": "<<FINAL_ANSWER>>\nSWE-agent answer\n<<FINAL_ANSWER>>",
            "extra_field": "preserved",
        },
        {"instance_id": "task-1"},
    )
    assert output["generation"] == "SWE-agent answer"
    assert output["instance_id"] == "task-1"
    assert output["model_name_or_path"] == "test-model"
    assert output["extra_field"] == "preserved"
    assert "model_patch" not in output


def test_swe_atlas_qna_maps_opencode_response_to_generation():
    task = object.__new__(SweAtlasQnAGenerationTask)
    task.cfg = SimpleNamespace(server=SimpleNamespace(model="test-model"))
    output = task._format_opencode_output(
        {
            "final_response": "<<FINAL_ANSWER>>\nOpenCode answer\n<<FINAL_ANSWER>>",
            "model_patch": None,
            "extra_field": "preserved",
        },
        {"instance_id": "task-1"},
    )
    assert output["generation"] == "OpenCode answer"
    assert output["instance_id"] == "task-1"
    assert output["model_name_or_path"] == "test-model"
    assert output["extra_field"] == "preserved"
    assert "final_response" not in output
    assert "model_patch" not in output


def test_swe_atlas_qna_generation_disables_inline_evaluation(monkeypatch, caplog):
    cfg = SimpleNamespace(
        agent_framework=SupportedAgentFrameworks.mini_swe_agent,
        agent_config=None,
        evaluate=True,
    )
    monkeypatch.setattr(SweBenchGenerationTask, "__init__", lambda self, cfg: None)

    SweAtlasQnAGenerationTask(cfg)

    assert cfg.evaluate is False
    assert cfg.agent_config == "eval/swe-atlas-qna/mini-swe-agent/default"
    assert "overriding evaluate=True with evaluate=False" in caplog.text


def test_swe_atlas_qna_selects_swe_agent_config(monkeypatch):
    cfg = SimpleNamespace(
        agent_framework=SupportedAgentFrameworks.swe_agent,
        agent_config=None,
        evaluate=False,
    )
    monkeypatch.setattr(SweBenchGenerationTask, "__init__", lambda self, cfg: None)

    SweAtlasQnAGenerationTask(cfg)

    assert cfg.agent_config == "eval/swe-atlas-qna/swe-agent/default"


def test_swe_atlas_qna_selects_opencode_config(monkeypatch):
    cfg = SimpleNamespace(
        agent_framework=SupportedAgentFrameworks.opencode,
        agent_config=None,
        evaluate=False,
    )
    monkeypatch.setattr(SweBenchGenerationTask, "__init__", lambda self, cfg: None)

    SweAtlasQnAGenerationTask(cfg)

    assert cfg.agent_config == "eval/swe-atlas-qna/opencode/default"


def test_swe_atlas_qna_swe_agent_prompt_submits_prose_answer():
    with open(get_config_path("eval/swe-atlas-qna/swe-agent/default"), encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    instance_template = config["agent"]["templates"]["instance_template"]
    bundles = config["agent"]["tools"]["bundles"]
    assert "Do NOT modify any files" in instance_template
    assert "cat <<'ANSWER_EOF' > /root/model.patch" in instance_template
    assert "echo '<<SWE_AGENT_SUBMISSION>>'" in instance_template
    assert bundles == [{"path": "tools/registry"}]


def test_swe_atlas_qna_opencode_prompt_is_read_only_qna():
    with open(get_config_path("eval/swe-atlas-qna/opencode/default"), encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    assert config["permission"]["bash"] == "allow"
    assert config["permission"]["edit"] == "deny"
    prompt = config["agent"]["build"]["prompt"]
    assert "do not modify repository files" in prompt
    assert "<<FINAL_ANSWER>>" in prompt


def test_swe_atlas_qna_processes_swe_agent_output(tmp_path):
    async def run():
        prediction_file = tmp_path / "prediction.jsonl"
        prediction_file.write_text(
            json.dumps({"model_patch": "<<FINAL_ANSWER>>\nFinal response\n<<FINAL_ANSWER>>"}),
            encoding="utf-8",
        )

        task = object.__new__(SweAtlasQnAGenerationTask)
        task.cfg = SimpleNamespace(
            agent_framework=SupportedAgentFrameworks.swe_agent,
            server=SimpleNamespace(model="test-model"),
        )
        task.semaphore = asyncio.Semaphore(1)
        task.get_api_base = lambda: "http://127.0.0.1:8000/v1"

        async def fake_run_swe_agent(data_point, api_base):
            assert data_point["instance_id"] == "task-1"
            assert data_point["base_commit"] == "test-commit"
            assert api_base == "http://127.0.0.1:8000/v1"
            return str(prediction_file)

        task._run_swe_agent = fake_run_swe_agent
        return await task.process_single_datapoint({"instance_id": "task-1", "base_commit": "test-commit"}, [])

    output = asyncio.run(run())

    assert output["generation"] == "Final response"
    assert output["swe-atlas-qna-outputs"]["instance_id"] == "task-1"


def test_swe_atlas_qna_processes_opencode_output(tmp_path):
    async def run():
        prediction_file = tmp_path / "prediction.jsonl"
        prediction_file.write_text(
            json.dumps(
                {
                    "final_response": "<<FINAL_ANSWER>>\nFinal response\n<<FINAL_ANSWER>>",
                    "model_patch": None,
                }
            ),
            encoding="utf-8",
        )

        task = object.__new__(SweAtlasQnAGenerationTask)
        task.cfg = SimpleNamespace(
            agent_framework=SupportedAgentFrameworks.opencode,
            server=SimpleNamespace(model="test-model"),
        )
        task.semaphore = asyncio.Semaphore(1)
        task.get_api_base = lambda: "http://127.0.0.1:8000/v1"

        async def fake_run_opencode(data_point, api_base):
            assert data_point["instance_id"] == "task-1"
            assert api_base == "http://127.0.0.1:8000/v1"
            return str(prediction_file)

        task._run_opencode = fake_run_opencode
        return await task.process_single_datapoint({"instance_id": "task-1"}, [])

    output = asyncio.run(run())

    assert output["generation"] == "Final response"
    assert output["swe-atlas-qna-outputs"]["instance_id"] == "task-1"


def test_opencode_config_injects_model_and_sampling_parameters():
    config = build_opencode_config(
        agent_config={"permission": {"edit": "deny"}},
        api_base="http://127.0.0.1:8000/v1",
        model="org/model",
        context_window=32768,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        extra_body={"chat_template_kwargs": {"enable_thinking": True}},
        agent_max_turns=25,
        tokens_to_generate=4096,
    )

    provider = config["provider"]["nemo"]
    model = provider["models"]["org/model"]
    assert provider["options"]["baseURL"] == "http://127.0.0.1:8000/v1"
    assert model["limit"] == {"context": 32768, "output": 4096}
    assert model["options"]["top_k"] == 40
    assert model["options"]["chat_template_kwargs"]["enable_thinking"] is True
    assert config["agent"]["build"] == {"temperature": 0.7, "top_p": 0.9, "steps": 25}
    assert config["permission"]["edit"] == "deny"


def test_opencode_config_clamps_default_output_to_context_window():
    config = build_opencode_config(
        agent_config={},
        api_base="http://127.0.0.1:8000/v1",
        model="test-model",
        context_window=32768,
        temperature=0.0,
        top_p=0.95,
        top_k=None,
        extra_body={},
        agent_max_turns=25,
    )

    assert config["provider"]["nemo"]["models"]["test-model"]["limit"] == {
        "context": 32768,
        "output": 32768,
    }


def test_opencode_config_rejects_null_extra_body_with_clear_error():
    with pytest.raises(
        ValueError,
        match=r"OpenCode inference\.extra_body cannot be null; omit it or set it to \{\}\.",
    ):
        build_opencode_config(
            agent_config={},
            api_base="http://127.0.0.1:8000/v1",
            model="test-model",
            context_window=32768,
            temperature=0.0,
            top_p=0.95,
            top_k=40,
            extra_body=None,
            agent_max_turns=25,
        )


def test_opencode_installer_selects_native_musl_package_for_alpine():
    command = build_opencode_install_command("1.17.11")

    assert "opencode-linux-arm64-musl" in command
    assert "opencode-linux-x64-baseline-musl" in command
    assert "${OPENCODE_MUSL_PACKAGE}@1.17.11" in command
    assert "ln -sf opencode-native /root/opencode/bin/opencode" in command
    assert "ld-musl-${OPENCODE_MUSL_ARCH}.so.1" in command
    assert "libstdc++.so.6" in command
    assert "node-v${NODE_VERSION}-${NODE_ARCH}.tar.gz" in command


def test_opencode_extracts_last_assistant_response_and_jsonl_fallback(tmp_path):
    session = {
        "messages": [
            {"info": {"role": "assistant"}, "parts": [{"type": "text", "text": "Earlier"}]},
            {"info": {"role": "user"}, "parts": [{"type": "text", "text": "Continue"}]},
            {
                "info": {"role": "assistant", "finish": "tool-calls"},
                "parts": [{"type": "text", "text": "Calling another tool"}],
            },
            {
                "info": {"role": "assistant", "finish": "stop"},
                "parts": [
                    {"type": "reasoning", "text": "Private reasoning"},
                    {"type": "text", "text": "Final answer"},
                ],
            },
        ]
    }
    assert extract_final_assistant_text(session) == "Final answer"

    event_file = tmp_path / "opencode.txt"
    event_file.write_text(
        "\n".join(
            [
                json.dumps({"type": "text", "part": {"messageID": "first", "id": "1", "text": "Earlier"}}),
                "not json",
                json.dumps({"type": "text", "part": {"messageID": "final", "id": "1", "text": "Fallback"}}),
                json.dumps({"type": "text", "part": {"messageID": "final", "id": "2", "text": "answer"}}),
            ]
        ),
        encoding="utf-8",
    )
    assert extract_final_assistant_text_from_jsonl(event_file) == "Fallback\nanswer"


def test_swe_atlas_qna_final_answer_extraction_falls_back_to_plain_submission():
    assert extract_final_answer("  Plain final answer  ") == "Plain final answer"
    assert extract_final_answer(None) == ""
