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
from nemo_skills.inference.model.base import BaseModel
from nemo_skills.pipeline.generate import _create_job_unified
from nemo_skills.pipeline.utils import eval as eval_utils
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


def test_prepare_eval_commands_propagates_cli_with_sandbox_to_generation_cmd(monkeypatch):
    """Ensure `--with-sandbox` is treated as an override when building eval commands.

    Previously, if a benchmark had `REQUIRES_SANDBOX` unset and the user passed
    `--with-sandbox`, the sandbox sidecar was still launched because `add_task()`
    ORed the two flags together. But the generation command only used the
    benchmark flag, so it would not wait for the sandbox to be ready.

    This checks that eval command construction now applies the same OR logic
    throughout and passes `with_sandbox=True` into the generation command.
    """
    benchmark_args = eval_utils.BenchmarkArgs(
        name="aime25",
        input_file="/tmp/aime25.jsonl",
        generation_args="",
        judge_args="",
        judge_pipeline_args={},
        requires_sandbox=False,
        keep_mounts_for_sandbox=False,
        generation_module="nemo_skills.inference.generate",
        num_samples=0,
        num_chunks=None,
        eval_subfolder="eval-results/aime25",
    )

    monkeypatch.setattr(eval_utils, "add_default_args", lambda *args, **kwargs: [benchmark_args])
    monkeypatch.setattr(eval_utils.pipeline_utils, "get_remaining_jobs", lambda **kwargs: {None: [None]})
    monkeypatch.setattr(eval_utils.pipeline_utils, "should_get_random_port", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        eval_utils.pipeline_utils,
        "configure_client",
        lambda **kwargs: ({}, "http://localhost:8000", ""),
    )

    captured = {}

    def fake_get_generation_cmd(*args, **kwargs):
        captured["with_sandbox"] = kwargs["with_sandbox"]
        return "echo generation"

    monkeypatch.setattr(eval_utils.pipeline_utils, "get_generation_cmd", fake_get_generation_cmd)

    eval_utils.prepare_eval_commands(
        cluster_config={"executor": "none"},
        benchmarks_or_groups="aime25",
        split=None,
        num_jobs=1,
        starting_seed=0,
        output_dir="/tmp/out",
        num_chunks=None,
        chunk_ids=None,
        rerun_done=False,
        rerun_ratelimit_errors=False,
        server_parameters={
            "model": "test-model",
            "server_type": "openai",
            "server_address": "http://localhost:8000",
            "server_gpus": 0,
            "server_nodes": 1,
            "server_args": "",
            "server_entrypoint": None,
            "server_container": None,
        },
        extra_arguments="",
        data_dir=None,
        exclusive=False,
        with_sandbox=True,
        keep_mounts_for_sandbox=False,
        wandb_parameters=None,
        eval_requires_judge=False,
    )

    assert captured["with_sandbox"] is True
