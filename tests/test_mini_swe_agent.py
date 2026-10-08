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
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf

from nemo_skills.inference.eval.swe_atlas_qna import SweAtlasQnAGenerationTask
from nemo_skills.inference.eval.swebench import SweBenchInferenceConfig, transform_litellm_reasoning_request


def test_transform_litellm_reasoning_request_adds_reasoning_alias_recursively():
    request = {
        "model": "model",
        "messages": [
            {
                "role": "assistant",
                "reasoning_content": "thinking",
                "metadata": {"reasoning_content": "nested"},
            },
            {
                "role": "assistant",
                "reasoning_content": "legacy",
                "reasoning": "authoritative",
            },
        ],
    }

    transformed = transform_litellm_reasoning_request(request)

    assert transformed["messages"][0]["reasoning"] == "thinking"
    assert transformed["messages"][0]["metadata"]["reasoning"] == "nested"
    assert transformed["messages"][1]["reasoning"] == "authoritative"
    assert "reasoning" not in request["messages"][0]


def test_transform_litellm_reasoning_request_promotes_provider_reasoning():
    request = {
        "messages": [
            {
                "role": "assistant",
                "content": "",
                "reasoning_content": "",
                "provider_specific_fields": {
                    "reasoning": "thinking from LiteLLM",
                    "refusal": None,
                },
            },
            {
                "role": "assistant",
                "reasoning": "authoritative",
                "provider_specific_fields": {"reasoning": "provider fallback"},
            },
            {
                "role": "user",
                "provider_specific_fields": {"reasoning": "not assistant reasoning"},
            },
        ]
    }

    transformed = transform_litellm_reasoning_request(request)

    assert transformed["messages"][0]["reasoning"] == "thinking from LiteLLM"
    assert transformed["messages"][1]["reasoning"] == "authoritative"
    assert "reasoning" not in transformed["messages"][2]
    assert "reasoning" not in request["messages"][0]


def test_transform_litellm_reasoning_request_leaves_non_message_fields_unchanged():
    request = {
        "reasoning_content": "top-level",
        "messages": [{"role": "user", "content": "hello"}],
    }

    transformed = transform_litellm_reasoning_request(request)

    assert transformed == request
    assert transformed is not request


def test_mini_swe_agent_creates_runtime_config_inside_staging_mount(tmp_path):
    task = object.__new__(SweAtlasQnAGenerationTask)
    task.output_dir = tmp_path
    inference = SweBenchInferenceConfig()
    inference.extra_body = OmegaConf.create({})
    task.cfg = SimpleNamespace(
        agent_config="eval/swe-atlas-qna/mini-swe-agent/default",
        agent_cwd=None,
        agent_max_turns=10,
        inference=inference,
        server=SimpleNamespace(model="test-model"),
    )
    task._get_extra_instructions = lambda: ""

    async def fake_execute(data_point, command_builder, expected_file_pattern, **kwargs):
        command = command_builder("http://proxy:1234/v1")
        assert "mkdir -p /trajectories_mount/configs" in command
        assert ">/trajectories_mount/configs/config_task-1.yaml" in command
        assert "http://proxy:1234/v1" in command

        trajectory_file = tmp_path / "trajectories" / "task-1.traj.json"
        trajectory_file.parent.mkdir()
        trajectory_file.write_text(
            json.dumps({"info": {"submission": "<<FINAL_ANSWER>>answer<<FINAL_ANSWER>>"}}),
            encoding="utf-8",
        )
        assert expected_file_pattern == str(trajectory_file)
        return str(trajectory_file)

    task._execute_agent_command_with_capture = fake_execute
    prediction_file = asyncio.run(
        task._run_mini_swe_agent(
            {"instance_id": "task-1", "problem_statement": "Question?"},
            "http://server/v1",
        )
    )

    prediction = json.loads(Path(prediction_file).read_text(encoding="utf-8"))
    assert prediction["generation"] == "answer"
