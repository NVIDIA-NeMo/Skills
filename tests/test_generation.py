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

# running most things through subprocess since that's how it's usually used
import subprocess
from types import SimpleNamespace

import pytest

from nemo_skills.evaluation.metrics import ComputeMetrics
from nemo_skills.inference.generate import GenerationTask, GenerationTaskConfig, InferenceConfig
from nemo_skills.inference.model.base import BaseModel, EndpointType
from nemo_skills.pipeline.generate import _create_job_unified
from nemo_skills.pipeline.utils.generation import configure_client
from nemo_skills.pipeline.utils.scripts import ServerScript


@pytest.mark.timeout(300)
def test_eval_gsm8k_api(tmp_path):
    cmd = (
        f"ns eval "
        f"    --server_type=openai "
        f"    --model=nvidia/nemotron-3-nano-30b-a3b "
        f"    --server_address=https://integrate.api.nvidia.com/v1 "
        f"    --benchmarks=gsm8k "
        f"    --output_dir={tmp_path} "
        f"    ++max_samples=2 "
        f"    ++max_concurrent_requests=1 "
        f"    ++inference.timeout=120 "
        f"    ++server.max_retries=1 "
    )
    subprocess.run(cmd, shell=True, check=True)

    # checking that summarize results works (just that there are no errors, but can inspect the output as well)
    subprocess.run(
        f"ns summarize_results {tmp_path}",
        shell=True,
        check=True,
    )

    # running compute_metrics to check that results are expected
    metrics = ComputeMetrics(benchmark="gsm8k").compute_metrics(
        [f"{tmp_path}/eval-results/gsm8k/output.jsonl"],
    )["_all_"]["pass@1"]

    assert metrics["symbolic_correct"] >= 80


@pytest.mark.timeout(300)
def test_eval_judge_api(tmp_path):
    cmd = (
        f"ns eval "
        f"    --server_type=openai "
        f"    --model=nvidia/nemotron-3-nano-30b-a3b "
        f"    --server_address=https://integrate.api.nvidia.com/v1 "
        f"    --benchmarks=math-500 "
        f"    --output_dir={tmp_path} "
        f"    --judge_model=nvidia/nemotron-3-nano-30b-a3b "
        f"    --judge_server_address=https://integrate.api.nvidia.com/v1 "
        f"    --judge_server_type=openai "
        f"    --judge_generation_type=math_judge "
        f"    --extra_judge_args='++max_concurrent_requests=1 ++inference.timeout=120 ++server.max_retries=1' "
        f"    ++max_samples=2 "
        f"    ++max_concurrent_requests=1 "
        f"    ++inference.timeout=120 "
        f"    ++server.max_retries=1 "
    )
    subprocess.run(cmd, shell=True, check=True)

    # checking that summarize results works (just that there are no errors, but can inspect the output as well)
    subprocess.run(
        f"ns summarize_results {tmp_path}",
        shell=True,
        check=True,
    )

    # running compute_metrics to check that results are expected
    metrics = ComputeMetrics(benchmark="math-500").compute_metrics(
        [f"{tmp_path}/eval-results/math-500/output.jsonl"],
    )["_all_"]["pass@1"]

    assert metrics["symbolic_correct"] >= 40
    assert metrics["judge_correct"] >= 40


def test_fail_on_api_key_env_var(tmp_path):
    cmd = (
        f"ns eval "
        f"    --server_type=openai "
        f"    --model=nvidia/nemotron-3-nano-30b-a3b "
        f"    --server_address=https://integrate.api.nvidia.com/v1 "
        f"    --benchmarks=gsm8k "
        f"    --output_dir={tmp_path} "
        f"    ++max_samples=2 "
        f"    ++server.api_key_env_var=MY_CUSTOM_KEY "
    )
    result = subprocess.run(cmd, shell=True, check=True, capture_output=True)

    # nemo-run always finishes with 0 error code, so just checking that expected exception is in the output
    assert (
        "ValueError: You defined api_key_env_var=MY_CUSTOM_KEY but the value is not set" in result.stdout.decode()
    ), result.stdout.decode()


@pytest.mark.timeout(300)
def test_succeed_on_api_key_env_var(tmp_path):
    cmd = (
        f"export MY_CUSTOM_KEY=$NVIDIA_API_KEY && "
        f"unset NVIDIA_API_KEY && "
        f"ns eval "
        f"    --server_type=openai "
        f"    --model=nvidia/nemotron-3-nano-30b-a3b "
        f"    --server_address=https://integrate.api.nvidia.com/v1 "
        f"    --benchmarks=gsm8k "
        f"    --output_dir={tmp_path} "
        f"    ++max_samples=2 "
        f"    ++max_concurrent_requests=1 "
        f"    ++inference.timeout=120 "
        f"    ++server.max_retries=1 "
        f"    ++server.api_key_env_var=MY_CUSTOM_KEY "
    )
    subprocess.run(cmd, shell=True, check=True)

    # checking that summarize results works (just that there are no errors, but can inspect the output as well)
    subprocess.run(
        f"ns summarize_results {tmp_path}",
        shell=True,
        check=True,
    )

    # running compute_metrics to check that results are expected
    metrics = ComputeMetrics(benchmark="gsm8k").compute_metrics(
        [f"{tmp_path}/eval-results/gsm8k/output.jsonl"],
    )["_all_"]["pass@1"]

    assert metrics["symbolic_correct"] >= 80


@pytest.mark.timeout(300)
@pytest.mark.parametrize("format", ["list", "dict"])
def test_generate_openai_format(tmp_path, format):
    cmd = (
        f"ns generate "
        f"    --server_type=openai "
        f"    --model=nvidia/nemotron-3-nano-30b-a3b "
        f"    --server_address=https://integrate.api.nvidia.com/v1 "
        f"    --input_file=/nemo_run/code/tests/data/openai-input-{format}.test "
        f"    --output_dir={tmp_path} "
        f"    ++prompt_format=openai "
        f"    ++max_concurrent_requests=1 "
        f"    ++inference.timeout=120 "
        f"    ++server.max_retries=1 "
    )
    subprocess.run(cmd, shell=True, check=True)

    # checking that output exists and has the expected format
    with open(f"{tmp_path}/output.jsonl") as fin:
        data = [json.loads(line) for line in fin.readlines()]
    assert len(data) == 2
    assert len(data[0]["generation"]) > 0
    assert len(data[1]["generation"]) > 0


def test_server_metadata_from_num_tasks(tmp_path):
    """Test that metadata dict is properly created from server command returning (cmd, num_tasks)."""
    cluster_config = {
        "containers": {
            "vllm": "apitest/vllm",
            "nemo-skills": "apitest/nemo-skills",
            "sandbox": "apitest/sandbox",
        },
        "executor": "none",
    }
    server_config = {
        "server_type": "vllm",
        "num_gpus": 8,
        "num_nodes": 1,
        "model_path": str(tmp_path / "model"),
        "server_port": 5000,
        "server_args": "",
    }
    generation_params = {"output_dir": "/tmp/out"}

    groups = _create_job_unified(
        models=[server_config["model_path"]],
        server_configs=[server_config],
        generation_params=generation_params,
        cluster_config=cluster_config,
        installation_command=None,
        with_sandbox=False,
        partition=None,
        account=None,
        keep_mounts_for_sandbox=False,
        task_name="test-task",
        log_dir="/tmp/logs",
    )

    server_cmd = groups[0].commands[0]
    assert isinstance(server_cmd.script, ServerScript)
    assert server_cmd.script.num_tasks >= 1
    assert server_cmd.script.num_gpus == server_config["num_gpus"]
    assert groups[0].hardware.num_gpus == server_config["num_gpus"]
    assert groups[0].hardware.num_tasks == server_cmd.script.num_tasks


class TokenizerProbeModel(BaseModel):
    def __init__(self, **kwargs):
        self.tokenizer_requests = []
        super().__init__(model="dummy-model", base_url="", **kwargs)

    def _get_tokenizer(self, tokenizer):
        self.tokenizer_requests.append(tokenizer)
        return "tokenizer"

    def _build_chat_request_params(self, **kwargs):
        return {}

    def _build_completion_request_params(self, **kwargs):
        return {}


@pytest.mark.parametrize(
    "enable_soft_fail,context_limit_retry_strategy,require_tokenizer,expected_requests",
    [
        (False, None, False, 0),
        (True, None, False, 0),
        (True, "reduce_generation", False, 0),
        (False, "reduce_prompt_from_start", False, 0),
        (True, "reduce_prompt_from_start", False, 1),
        (True, "reduce_prompt_from_end", False, 1),
        (False, None, True, 1),
    ],
)
def test_base_model_initializes_tokenizer_only_when_needed(
    enable_soft_fail,
    context_limit_retry_strategy,
    require_tokenizer,
    expected_requests,
):
    model = TokenizerProbeModel(
        tokenizer="explicit-tokenizer",
        enable_soft_fail=enable_soft_fail,
        context_limit_retry_strategy=context_limit_retry_strategy,
        require_tokenizer=require_tokenizer,
    )

    assert len(model.tokenizer_requests) == expected_requests
    assert (model.tokenizer == "tokenizer") == (expected_requests == 1)


@pytest.mark.parametrize(
    "server_overrides,expected_tokenizer",
    [
        ({}, None),
        ({"enable_soft_fail": True}, None),
        ({"enable_soft_fail": True, "context_limit_retry_strategy": "reduce_generation"}, None),
        ({"enable_soft_fail": False, "context_limit_retry_strategy": "reduce_prompt_from_start"}, None),
        (
            {"enable_soft_fail": True, "context_limit_retry_strategy": "reduce_prompt_from_start"},
            "explicit-tokenizer",
        ),
        ({"enable_soft_fail": True, "context_limit_retry_strategy": "reduce_prompt_from_end"}, "explicit-tokenizer"),
    ],
)
def test_generation_task_sets_tokenizer_only_for_prompt_retry_strategies(
    monkeypatch,
    server_overrides,
    expected_tokenizer,
):
    monkeypatch.setattr(GenerationTask, "setup_litellm_cache", lambda self: None)
    monkeypatch.setattr(GenerationTask, "setup_prompt", lambda self: None)
    monkeypatch.setattr(GenerationTask, "setup_llm", lambda self: object())

    cfg = GenerationTaskConfig(
        input_file="input.jsonl",
        output_file="output.jsonl",
        prompt_format="openai",
        tokenizer="explicit-tokenizer",
        server={"server_type": "openai", "model": "server-model", **server_overrides},
    )

    task = GenerationTask(cfg)

    assert task.tokenizer == expected_tokenizer


def test_generation_task_keeps_text_endpoint_tokenizer(monkeypatch):
    monkeypatch.setattr(GenerationTask, "setup_litellm_cache", lambda self: None)
    monkeypatch.setattr(GenerationTask, "setup_prompt", lambda self: None)
    monkeypatch.setattr(GenerationTask, "setup_llm", lambda self: object())

    cfg = GenerationTaskConfig(
        input_file="input.jsonl",
        output_file="output.jsonl",
        prompt_format="openai",
        tokenizer="explicit-tokenizer",
        server={"server_type": "openai", "model": "server-model"},
        inference=InferenceConfig(endpoint_type=EndpointType.text, tokens_to_generate=1),
    )

    task = GenerationTask(cfg)

    assert task.tokenizer == "explicit-tokenizer"


@pytest.mark.parametrize(
    "server_nodes,expected_host",
    [
        (1, "127.0.0.1"),
        (2, "$SLURM_MASTER_NODE"),
    ],
)
def test_configure_client_hosted_server_host_depends_on_num_nodes(server_nodes, expected_host):
    server_config, server_address, extra_arguments = configure_client(
        model="/models/test-model",
        server_type="vllm",
        server_address=None,
        server_gpus=8,
        server_nodes=server_nodes,
        server_args="",
        server_entrypoint=None,
        get_random_port=False,
        extra_arguments="++foo=bar",
    )

    assert server_config["server_port"] == 5000
    assert server_address == "localhost:5000"
    assert f"++server.host={expected_host}" in extra_arguments
    assert "++server.port=5000" in extra_arguments
    assert "++server.model=/models/test-model" in extra_arguments
    assert extra_arguments.count("++server.server_type=") == 1
    assert "++server.server_type=vllm" in extra_arguments


def test_configure_client_preserves_explicit_server_type_override():
    _, _, extra_arguments = configure_client(
        model="/models/test-model",
        server_type="vllm",
        server_address=None,
        server_gpus=8,
        server_nodes=2,
        server_args="",
        server_entrypoint=None,
        get_random_port=False,
        extra_arguments="++server.server_type=vllm_multimodal",
    )

    assert extra_arguments.count("++server.server_type=") == 1
    assert "++server.server_type=vllm_multimodal" in extra_arguments
    assert "++server.host=$SLURM_MASTER_NODE" in extra_arguments


@pytest.mark.timeout(300)
def test_judge_generations_with_structured_output(tmp_path):
    cmd = (
        f"ns eval "
        f"    --server_type=openai "
        f"    --model=nvidia/nemotron-3-nano-30b-a3b "
        f"    --server_address=https://integrate.api.nvidia.com/v1 "
        f"    --benchmarks=hle "
        f"    --output_dir={tmp_path} "
        f"    --judge_model=nvidia/nemotron-3-nano-30b-a3b "
        f"    --judge_server_address=https://integrate.api.nvidia.com/v1 "
        f"    --judge_server_type=openai "
        f"    --metric_type=hle-aa "
        f'    --extra_judge_args="++structured_output=HLE_JUDGE_AA ++max_concurrent_requests=1 ++inference.timeout=120 ++server.max_retries=1" '
        f"    ++max_samples=2 "
        f"    ++max_concurrent_requests=1 "
        f"    ++inference.timeout=120 "
        f"    ++server.max_retries=1 "
        f"    ++inference.tokens_to_generate=1024 "  # to make test go fast
    )
    subprocess.run(cmd, shell=True, check=True)

    # checking that output exists and has the expected format
    with open(f"{tmp_path}/eval-results/hle/output.jsonl") as fin:
        data = [json.loads(line) for line in fin.readlines()]
    judgements = [json.loads(data[i]["judgement"]) for i in range(len(data))]
    expected_keys = {"extracted_final_answer", "reasoning", "correct", "confidence"}
    assert set(judgements[0].keys()) == expected_keys
    assert set(judgements[1].keys()) == expected_keys
    assert judgements[0]["correct"] in {"yes", "no"}
    assert judgements[1]["correct"] in {"yes", "no"}


def test_process_chat_chunk_never_yields_none_generation():
    """Regression test for https://github.com/NVIDIA-NeMo/Skills/issues/1267

    OpenAI-compatible streaming APIs can emit chunks where delta.content is None.
    _process_chat_chunk must never return {"generation": None}.
    """
    model = BaseModel.__new__(BaseModel)
    p = model._process_chat_chunk

    def _chunk(content, finish_reason=None, *, use_delta=True):
        if use_delta:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        delta=SimpleNamespace(content=content),
                        finish_reason=finish_reason,
                    )
                ]
            )
        return SimpleNamespace(choices=[SimpleNamespace(text=content, finish_reason=finish_reason)])

    # Normal text
    assert p(_chunk("Hello"))[0]["generation"] == "Hello"

    # None delta (intermediate chunk) — was the bug
    assert p(_chunk(None))[0]["generation"] == ""

    # None on finish
    r = p(_chunk(None, finish_reason="stop"))[0]
    assert r["generation"] == "" and r["finish_reason"] == "stop"

    # Non-delta (completion-style) with text=None
    assert p(_chunk(None, use_delta=False))[0]["generation"] == ""

    # Full consumer loop — must not raise TypeError
    full = ""
    for c in [
        _chunk("Hello "),
        _chunk(None),
        _chunk("world"),
        _chunk(None, finish_reason="stop"),
    ]:
        for r in p(c):
            full += r["generation"]
    assert full == "Hello world"


@pytest.mark.parametrize(
    "usage_kwargs,expected_input",
    [
        ({"prompt_tokens": 5}, 5),
        ({"input_tokens": 7}, 7),
        ({}, None),
    ],
)
def test_parse_completion_response_token_counts(usage_kwargs, expected_input):
    model = BaseModel.__new__(BaseModel)
    usage = SimpleNamespace(completion_tokens=10, **usage_kwargs)
    response = SimpleNamespace(
        usage=usage,
        choices=[SimpleNamespace(text="hi", finish_reason="stop", logprobs=None)],
    )
    result = model._parse_completion_response(response)
    assert result["num_generated_tokens"] == 10
    assert result.get("num_input_tokens") == expected_input
