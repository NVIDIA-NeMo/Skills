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

from copy import deepcopy
from types import SimpleNamespace

import pytest

from nemo_skills.inference.prover import ProverTask

DATA_POINT = {
    "header": "",
    "informal_prefix": "theorem t : True := by sorry",
    "formal_statement": "",
}


class FakePrompt:
    def __init__(self, messages=None):
        self.messages = messages or [{"role": "user", "content": "base prompt"}]

    def fill(self, _):
        return deepcopy(self.messages)


class FakeRefinePrompt:
    def fill(self, data):
        return [{"role": "user", "content": f"feedback: {data['error_message'][:40]}"}]


class RecordingTokenizer:
    def __init__(self, token_count_fn=None):
        self.token_count_fn = token_count_fn or (lambda conv: max(1, len(conv) * 5))
        self.conversations = []

    def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=True):  # noqa: ARG002
        if tokenize:
            self.conversations.append(deepcopy(conversation))
            return [0] * self.token_count_fn(conversation)
        return "chat prompt"


class FakeSandbox:
    def __init__(self, result):
        self.result = result

    async def execute_code(self, *args, **kwargs):  # noqa: ARG002
        return self.result, None


def build_task(
    *,
    refinement=True,
    refinement_max_turns=3,
    nemotron_refinement=False,
    remove_cot=True,
    delete_wrong_turns=False,
    max_tokens=40960,
    tokenizer=None,
    sandbox=None,
):
    task = object.__new__(ProverTask)
    task.cfg = SimpleNamespace(
        refinement=refinement,
        refinement_max_turns=refinement_max_turns,
        nemotron_refinement=nemotron_refinement,
        max_tokens=max_tokens,
        inference=SimpleNamespace(tokens_to_generate=64),
        remove_cot=remove_cot,
        delete_wrong_turns=delete_wrong_turns,
    )
    task.prompt = FakePrompt()
    task.hf_tokenizer = tokenizer or RecordingTokenizer()
    task.sandbox = sandbox
    if refinement:
        task.refine_prompt = FakeRefinePrompt()
    return task


@pytest.mark.asyncio
async def test_parse_failure_retries_cleanly_from_base_prompt():
    tokenizer = RecordingTokenizer()
    task = build_task(refinement=True, refinement_max_turns=3, nemotron_refinement=False, tokenizer=tokenizer)
    calls = {"count": 0}

    async def fake_generate(prompt, **kwargs):  # noqa: ARG001
        calls["count"] += 1
        return {"generation": "I forgot the fenced lean block."}

    async def fake_extract(formal_statement, generation):  # noqa: ARG001
        return "None", "None"

    task._generate_single_completion = fake_generate
    task._extract_and_replace_code = fake_extract

    result = await ProverTask._single_data_point_generate(task, DATA_POINT, data=[])

    assert calls["count"] == 3
    assert len(result["results_dict_list"]) == 3
    assert all(turn["success"] is False for turn in result["results_dict_list"])
    assert all("feedback" not in turn for turn in result["results_dict_list"])
    assert all(conv == [{"role": "user", "content": "base prompt"}] for conv in tokenizer.conversations)
    assert result["prompt_turn_list"] == [{"role": "user", "content": "base prompt"}]


@pytest.mark.asyncio
async def test_parse_failure_does_not_use_nemotron_refinement_state():
    tokenizer = RecordingTokenizer()
    task = build_task(refinement=True, refinement_max_turns=3, nemotron_refinement=True, tokenizer=tokenizer)
    transform_calls = []

    async def fake_generate(prompt, **kwargs):  # noqa: ARG001
        return {"generation": "still not in fenced code format"}

    async def fake_extract(formal_statement, generation):  # noqa: ARG001
        return "None", "None"

    def fake_transform(proof_attempt, error_message):
        transform_calls.append((proof_attempt, error_message))
        return [{"role": "user", "content": "nemotron transformed prompt"}]

    task._generate_single_completion = fake_generate
    task._extract_and_replace_code = fake_extract
    task._transform_for_nemotron_refinement = fake_transform

    await ProverTask._single_data_point_generate(task, DATA_POINT, data=[])

    assert transform_calls == []
    assert all(conv == [{"role": "user", "content": "base prompt"}] for conv in tokenizer.conversations)


@pytest.mark.asyncio
async def test_compile_timeout_keeps_refinement_feedback_retries():
    tokenizer = RecordingTokenizer()
    sandbox = FakeSandbox({"process_status": "timeout", "stdout": "", "stderr": ""})
    task = build_task(
        refinement=True,
        refinement_max_turns=3,
        nemotron_refinement=False,
        remove_cot=False,
        tokenizer=tokenizer,
        sandbox=sandbox,
    )
    calls = {"count": 0}

    async def fake_generate(prompt, **kwargs):  # noqa: ARG001
        calls["count"] += 1
        return {"generation": f"```lean4\n-- attempt {calls['count']}\n```"}

    async def fake_extract(formal_statement, generation):  # noqa: ARG001
        return "some_code", "theorem t : True := by\n  trivial"

    task._generate_single_completion = fake_generate
    task._extract_and_replace_code = fake_extract

    result = await ProverTask._single_data_point_generate(task, DATA_POINT, data=[])

    assert calls["count"] == 3
    assert len(result["results_dict_list"]) == 3
    assert all(turn["success"] is False for turn in result["results_dict_list"])
    assert all("feedback" in turn for turn in result["results_dict_list"])
    assert [len(conv) for conv in tokenizer.conversations] == [1, 3, 5]


@pytest.mark.asyncio
async def test_remove_cot_delete_wrong_turns_keeps_only_latest_clean_code_on_success():
    sandbox = FakeSandbox({"process_status": "completed", "stdout": "", "stderr": ""})
    task = build_task(
        refinement=False,
        refinement_max_turns=2,
        remove_cot=True,
        delete_wrong_turns=True,
        sandbox=sandbox,
    )

    async def fake_generate(prompt, **kwargs):  # noqa: ARG001
        return {"generation": "raw model response"}

    async def fake_extract(formal_statement, generation):  # noqa: ARG001
        return "some_code", "theorem t : True := by\n  trivial"

    task._generate_single_completion = fake_generate
    task._extract_and_replace_code = fake_extract

    result = await ProverTask._single_data_point_generate(task, DATA_POINT, data=[])

    assert result["success"] is True
    assert len(result["results_dict_list"]) == 1
    assert len(result["prompt_turn_list"]) == 2
    assert result["prompt_turn_list"][1]["role"] == "assistant"
    assert "```lean4" in result["prompt_turn_list"][1]["content"]
    assert "theorem t : True := by" in result["prompt_turn_list"][1]["content"]
    assert len(result["full_prompt_turn_list"]) == 2
    assert result["full_prompt_turn_list"][1]["content"] == "raw model response"


@pytest.mark.asyncio
async def test_prefix_too_long_exits_before_generation():
    tokenizer = RecordingTokenizer(token_count_fn=lambda conv: 50)
    task = build_task(refinement=True, refinement_max_turns=4, max_tokens=10, tokenizer=tokenizer)
    calls = {"count": 0}

    async def fake_generate(prompt, **kwargs):  # noqa: ARG001
        calls["count"] += 1
        return {"generation": "should not be called"}

    async def fake_extract(formal_statement, generation):  # noqa: ARG001
        return "None", "None"

    task._generate_single_completion = fake_generate
    task._extract_and_replace_code = fake_extract

    result = await ProverTask._single_data_point_generate(task, DATA_POINT, data=[])

    assert calls["count"] == 0
    assert result["results_dict_list"] == []
    assert result["success"] is False


def test_parse_gpt_oss_output_extracts_channels():
    task = object.__new__(ProverTask)
    content = (
        "<|channel|>analysis<|message|>step-by-step<|end|>"
        "<|start|>assistant<|channel|>final<|message|>final answer<|return|>"
    )

    final_content, thinking = ProverTask._parse_gpt_oss_output(task, content)

    assert final_content == "final answer"
    assert thinking == "step-by-step"
