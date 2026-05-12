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

import json
import os
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import nemo_skills.pipeline.utils.scripts.eval as eval_scripts
from nemo_skills.pipeline import eval as eval_pipeline
from nemo_skills.pipeline.utils import eval as eval_utils
from nemo_skills.pipeline.utils.scripts import EvalClientScript


class FakeExp:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_eval_client_script_parallel_fails_if_any_unit_fails(monkeypatch, tmp_path):
    """Ensure eval units fail if any command fails in parallel mode."""
    failed_marker = tmp_path / "failed.txt"
    succeeded_marker = tmp_path / "succeeded.txt"

    monkeypatch.setattr(eval_scripts, "get_generation_cmd", lambda *, command, **kwargs: command)
    monkeypatch.setattr(eval_scripts, "wrap_python_path", lambda cmd: cmd)

    units = [
        {"command": f"echo failed > {shlex.quote(str(failed_marker))}; sleep 0.1; exit 1"},
        {"command": f"echo succeeded > {shlex.quote(str(succeeded_marker))}; sleep 0.2; exit 0"},
    ]
    client_script = EvalClientScript(units=units, single_node_mode="parallel")

    cmd, _ = client_script.inline()
    result = subprocess.run(["bash", "-lc", cmd], check=False)

    assert result.returncode == 1
    assert failed_marker.read_text().strip() == "failed"
    assert succeeded_marker.read_text().strip() == "succeeded"


def test_prepare_eval_commands_propagates_cli_with_sandbox_to_generation_cmd(monkeypatch):
    """Ensure `--with-sandbox` is treated as an override when building eval commands.

    Previously, if a benchmark had `REQUIRES_SANDBOX` unset and the user passed
    `--with-sandbox`, the sandbox sidecar was still launched because `add_task()`
    ORed the two flags together. This checks that the prepared eval generation
    unit keeps `with_sandbox=True` all the way into `get_generation_cmd`.
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

    captured = {}

    def fake_get_generation_cmd(*args, **kwargs):
        captured["with_sandbox"] = kwargs["with_sandbox"]
        return "echo generation"

    monkeypatch.setattr("nemo_skills.pipeline.utils.scripts.eval.get_generation_cmd", fake_get_generation_cmd)

    _, job_batches = eval_utils.prepare_eval_commands(
        cluster_config={"executor": "none"},
        benchmarks_or_groups="aime25",
        split=None,
        num_jobs=1,
        starting_seed=0,
        output_dir="/tmp/out",
        num_chunks=None,
        chunk_ids=None,
        rerun_done=False,
        extra_arguments="",
        data_dir=None,
        exclusive=False,
        with_sandbox=True,
        keep_mounts_for_sandbox=False,
        wandb_parameters=None,
        eval_requires_judge=False,
    )

    units = [vars(unit).copy() for unit in job_batches[0][0]]
    client_script = EvalClientScript(units=units)
    client_script.inline()

    assert captured["with_sandbox"] is True


def test_resolve_child_sbatch_kwargs_inherits_or_overrides():
    parent = {"segment": 4, "qos": "batch"}

    assert eval_pipeline._resolve_child_sbatch_kwargs(parent, None) is parent
    assert eval_pipeline._resolve_child_sbatch_kwargs(parent, {}) is None
    assert eval_pipeline._resolve_child_sbatch_kwargs(parent, '{"segment": 1}') == {"segment": 1}


def _patch_eval_for_sbatch_tests(monkeypatch, benchmark_args):
    cluster_config = {"executor": "slurm", "containers": {"nemo-skills": "nemo-skills-container"}}

    monkeypatch.setattr(eval_pipeline.pipeline_utils, "get_cluster_config", lambda *args, **kwargs: cluster_config)
    monkeypatch.setattr(eval_pipeline.pipeline_utils, "resolve_mount_paths", lambda config, *args, **kwargs: config)
    monkeypatch.setattr(eval_pipeline.pipeline_utils, "get_env_variables", lambda config: {})
    monkeypatch.setattr(
        eval_pipeline.pipeline_utils,
        "check_mounts",
        lambda config, log_dir, mount_map, check_mounted_paths: (
            next(iter(mount_map.keys())),
            list(mount_map.keys())[1],
            log_dir,
        ),
    )
    monkeypatch.setattr(eval_pipeline.pipeline_utils, "get_exp", lambda *args, **kwargs: FakeExp())
    monkeypatch.setattr(eval_pipeline.pipeline_utils, "run_exp", lambda *args, **kwargs: None)
    monkeypatch.setattr(eval_pipeline, "prepare_eval_commands", lambda **kwargs: ({"gsm8k": benchmark_args()}, []))


@pytest.mark.parametrize(
    "summarize_sbatch_kwargs,expected_sbatch_kwargs",
    [
        (None, {"segment": 4, "qos": "main"}),
        ({}, None),
        ({"segment": 1}, {"segment": 1}),
    ],
)
def test_eval_summarize_sbatch_kwargs_and_account(
    monkeypatch, tmp_path, summarize_sbatch_kwargs, expected_sbatch_kwargs
):
    def benchmark_args():
        return eval_utils.BenchmarkArgs(
            name="gsm8k",
            input_file="/tmp/gsm8k.jsonl",
            generation_args="",
            judge_args="",
            judge_pipeline_args={},
            requires_sandbox=False,
            keep_mounts_for_sandbox=False,
            generation_module="nemo_skills.inference.generate",
            num_samples=0,
            num_chunks=None,
            eval_subfolder="eval-results/gsm8k",
            metrics_type="math",
        )

    _patch_eval_for_sbatch_tests(monkeypatch, benchmark_args)
    captured = []

    def fake_add_task(*args, **kwargs):
        captured.append(kwargs)
        return f"task-{len(captured)}"

    monkeypatch.setattr(eval_pipeline.pipeline_utils, "add_task", fake_add_task)

    eval_pipeline.eval(
        ctx=SimpleNamespace(args=[]),
        cluster="test-cluster",
        output_dir=str(tmp_path),
        benchmarks="gsm8k",
        model="model",
        server_type="openai",
        server_address="http://server",
        account="acct",
        sbatch_kwargs={"segment": 4, "qos": "main"},
        summarize_sbatch_kwargs=summarize_sbatch_kwargs,
    )

    assert len(captured) == 1
    assert captured[0]["account"] == "acct"
    assert captured[0]["sbatch_kwargs"] == expected_sbatch_kwargs


def test_eval_judge_sbatch_kwargs_override(monkeypatch, tmp_path):
    def benchmark_args():
        return eval_utils.BenchmarkArgs(
            name="gsm8k",
            input_file="/tmp/gsm8k.jsonl",
            generation_args="",
            judge_args="++judge=True",
            judge_pipeline_args={},
            requires_sandbox=False,
            keep_mounts_for_sandbox=False,
            generation_module="nemo_skills.inference.generate",
            num_samples=0,
            num_chunks=None,
            eval_subfolder="tmp-eval-results/gsm8k",
            metrics_type="math",
        )

    _patch_eval_for_sbatch_tests(monkeypatch, benchmark_args)
    captured = {}

    def fake_generate(**kwargs):
        captured["sbatch_kwargs"] = kwargs["sbatch_kwargs"]
        captured["account"] = kwargs["account"]
        return ["judge-task"]

    monkeypatch.setattr(eval_pipeline, "_generate", fake_generate)

    eval_pipeline.eval(
        ctx=SimpleNamespace(args=[]),
        cluster="test-cluster",
        output_dir=str(tmp_path),
        benchmarks="gsm8k",
        model="model",
        server_type="openai",
        server_address="http://server",
        account="acct",
        sbatch_kwargs={"segment": 4, "qos": "main"},
        judge_sbatch_kwargs={"segment": 1},
        auto_summarize_results=False,
    )

    assert captured["account"] == "acct"
    assert captured["sbatch_kwargs"] == {"segment": 1}


@pytest.mark.timeout(300)
@pytest.mark.skipif("NVIDIA_API_KEY" not in os.environ, reason="requires NVIDIA_API_KEY")
def test_eval_multi_model_generation_module_smoke(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "out"
    generation_module = repo_root / "tests" / "data" / "multi_model_eval_smoke.py"

    cmd = (
        f"ns eval "
        f"    --server_type=openai "
        f"    --server_type=openai "
        f"    --model=nvidia/nemotron-3-nano-30b-a3b "
        f"    --model=nvidia/nemotron-3-nano-30b-a3b "
        f"    --server_address=https://integrate.api.nvidia.com/v1 "
        f"    --server_address=https://integrate.api.nvidia.com/v1 "
        f"    --benchmarks=gsm8k "
        f"    --output_dir={shlex.quote(str(output_dir))} "
        f"    --generation_module={shlex.quote(str(generation_module))} "
        f"    ++max_samples=1 "
        f"    ++max_concurrent_requests=1 "
        f"    ++inference.timeout=120 "
        f"    ++server.max_retries=1 "
    )
    env = {**os.environ, "PYTHONPATH": f"{repo_root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    subprocess.run(cmd, shell=True, check=True, env=env)

    output_file = output_dir / "eval-results" / "gsm8k" / "output.jsonl"
    with output_file.open(encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]

    assert len(data) == 1
    assert data[0]["symbolic_correct"] is True
    assert data[0]["model_0_exact_match"] is True
    assert data[0]["model_1_exact_match"] is True
