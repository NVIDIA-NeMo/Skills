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
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

import nemo_skills.pipeline.utils.scripts.eval as eval_scripts
from nemo_skills.pipeline import eval as eval_pipeline
from nemo_skills.pipeline.utils import eval as eval_utils
from nemo_skills.pipeline.utils.scripts import BaseJobScript, EvalClientScript


class FakeExp:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@dataclass(kw_only=True)
class HostnameScript(BaseJobScript):
    port: int = 5000

    def __post_init__(self):
        self.set_inline("echo test")
        super().__post_init__()


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


def test_eval_client_script_uses_slurm_master_ref_for_self_hosted_single_group(monkeypatch):
    captured = {}

    def fake_get_generation_cmd(*args, **kwargs):
        captured["extra_arguments"] = kwargs["extra_arguments"]
        return "echo generation"

    monkeypatch.setattr(eval_scripts, "get_generation_cmd", fake_get_generation_cmd)
    monkeypatch.setattr(eval_scripts, "wrap_python_path", lambda cmd: cmd)

    server = HostnameScript(port=53587)
    client_script = EvalClientScript(
        units=[{"extra_arguments": ""}],
        servers=[server],
        model_names=["/models/test-model"],
        server_types=["vllm"],
    )

    client_script.inline()

    assert "++server.host=${SLURM_MASTER_NODE:-127.0.0.1}" in captured["extra_arguments"]
    assert "++server.port=53587" in captured["extra_arguments"]


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


@pytest.mark.parametrize(
    "ctx_args,alpine_container,expected_client_containers",
    [
        (
            [],
            "alpine-container",
            ["ubuntu-container"] * 6 + ["alpine-container"] * 4,
        ),
        (
            ["++execution_backend=opensandbox"],
            None,
            ["ubuntu-container"] * 10,
        ),
    ],
    ids=["apptainer", "opensandbox"],
)
@pytest.mark.parametrize("benchmark", ["swe-bench-pro", "swe-bench-pro-v2"])
@pytest.mark.parametrize("split", ["default.ubuntu,default.alpine", None], ids=["explicit-splits", "default-splits"])
@pytest.mark.parametrize("rerun_done", [False, True], ids=["normal", "rerun-done"])
def test_eval_expands_swe_bench_pro_splits_and_merges_before_summarizing(
    monkeypatch, tmp_path, ctx_args, alpine_container, expected_client_containers, benchmark, split, rerun_done
):
    def benchmark_args(split, suffix):
        return eval_utils.BenchmarkArgs(
            name=benchmark,
            input_file=f"/data/{benchmark}/{split}.jsonl",
            generation_args=f"++eval_config.split={split}",
            judge_args="",
            judge_pipeline_args={},
            requires_sandbox=False,
            keep_mounts_for_sandbox=False,
            generation_module=(
                "nemo_skills.inference.eval.swebench_pro_v2"
                if benchmark == "swe-bench-pro-v2"
                else "nemo_skills.inference.eval.swe_bench"
            ),
            num_samples=0,
            num_chunks=None,
            eval_subfolder=f"eval-results/{benchmark}/{suffix}",
            metrics_type="swe-bench",
        )

    _patch_eval_for_sbatch_tests(monkeypatch, lambda: benchmark_args("default.ubuntu", "unused"))

    def fake_prepare_eval_commands(*, split, num_chunks, eval_subfolder_suffix=None, **kwargs):
        args = benchmark_args(split, eval_subfolder_suffix)
        args.num_chunks = num_chunks
        batch_count = num_chunks or 1
        batches = []
        for chunk_id in range(batch_count):
            unit = eval_utils.EvalGenerationUnit(
                output_dir=str(tmp_path / args.eval_subfolder),
                input_file=args.input_file,
                extra_arguments=args.generation_args,
                random_seed=None,
                chunk_id=chunk_id if num_chunks else None,
                num_chunks=num_chunks,
                script=args.generation_module,
                requirements=[],
                wandb_parameters=None,
                with_sandbox=False,
            )
            batches.append(([unit], {benchmark}, False, False, []))
            args.job_ids.append(chunk_id)
        return {benchmark: args}, batches

    monkeypatch.setattr(eval_pipeline, "prepare_eval_commands", fake_prepare_eval_commands)
    monkeypatch.setattr(
        eval_pipeline,
        "_count_jsonl_rows",
        lambda cluster_config, input_file: 300 if "ubuntu" in input_file else 200,
    )

    submitted_pipelines = []

    def fake_pipeline_run(self, **kwargs):
        submitted_pipelines.append(self)
        return [f"generation-{idx}" for idx in range(len(self.jobs))]

    monkeypatch.setattr(eval_pipeline.Pipeline, "run", fake_pipeline_run)
    added_tasks = []

    def fake_add_task(*args, **kwargs):
        added_tasks.append(kwargs)
        return f"task-{len(added_tasks)}"

    monkeypatch.setattr(eval_pipeline.pipeline_utils, "add_task", fake_add_task)

    eval_pipeline.eval(
        ctx=SimpleNamespace(args=ctx_args),
        cluster="test-cluster",
        output_dir=str(tmp_path),
        benchmarks=benchmark,
        split=split,
        num_chunks=10,
        rerun_done=rerun_done,
        main_container="ubuntu-container",
        alpine_container="alpine-container",
        model="model",
        server_type="openai",
        server_address="http://server",
        skip_hf_home_check=True,
    )

    jobs = submitted_pipelines[0].jobs
    assert len(jobs) == 10
    client_containers = [job["group"].commands[-1].container for job in jobs]
    assert client_containers == expected_client_containers

    assert len(added_tasks) == 2
    merge_task, summarize_task = added_tasks
    assert "default.ubuntu" in merge_task["cmd"]
    assert "default.alpine" in merge_task["cmd"]
    assert ("if [ ! -f" in merge_task["cmd"]) is not rerun_done
    assert merge_task["container"] == "ubuntu-container"
    assert merge_task["task_dependencies"] == [f"generation-{idx}" for idx in range(10)]
    assert summarize_task["task_dependencies"] == ["task-1"]


@pytest.mark.parametrize("benchmark", ["swe-bench-pro", "swe-bench-pro-v2"])
def test_eval_requires_alpine_container_for_multi_split_swe_bench_pro(monkeypatch, tmp_path, benchmark):
    def benchmark_args():
        return eval_utils.BenchmarkArgs(
            name=benchmark,
            input_file=f"/data/{benchmark}/default.alpine.jsonl",
            generation_args="",
            judge_args="",
            judge_pipeline_args={},
            requires_sandbox=False,
            keep_mounts_for_sandbox=False,
            generation_module="nemo_skills.inference.eval.swe_bench",
            num_samples=0,
            num_chunks=None,
            eval_subfolder=f"eval-results/{benchmark}",
        )

    _patch_eval_for_sbatch_tests(monkeypatch, benchmark_args)

    with pytest.raises(ValueError, match="--alpine-container is required"):
        eval_pipeline.eval(
            ctx=SimpleNamespace(args=[]),
            cluster="test-cluster",
            output_dir=str(tmp_path),
            benchmarks=benchmark,
            split="default.ubuntu,default.alpine",
            model="model",
            server_type="openai",
            server_address="http://server",
        )


@pytest.mark.parametrize("benchmark", ["swe-bench-pro", "swe-bench-pro-v2"])
def test_eval_requires_container_for_single_alpine_swe_bench_pro(monkeypatch, tmp_path, benchmark):
    args = eval_utils.BenchmarkArgs(
        name=benchmark,
        input_file=f"/data/{benchmark}/default.alpine.jsonl",
        generation_args="",
        judge_args="",
        judge_pipeline_args={},
        requires_sandbox=False,
        keep_mounts_for_sandbox=False,
        generation_module="nemo_skills.inference.eval.swe_bench",
        num_samples=0,
        num_chunks=None,
        eval_subfolder=f"eval-results/{benchmark}",
    )
    _patch_eval_for_sbatch_tests(monkeypatch, lambda: args)

    with pytest.raises(ValueError, match="An Alpine NeMo-Skills container is required"):
        eval_pipeline.eval(
            ctx=SimpleNamespace(args=[]),
            cluster="test-cluster",
            output_dir=str(tmp_path),
            benchmarks=benchmark,
            split="default.alpine",
            model="model",
            server_type="openai",
            server_address="http://server",
        )


@pytest.mark.parametrize("benchmark", ["swe-bench-pro", "swe-bench-pro-v2"])
def test_eval_single_alpine_split_keeps_main_container_compatibility(monkeypatch, tmp_path, benchmark):
    args = eval_utils.BenchmarkArgs(
        name=benchmark,
        input_file=f"/data/{benchmark}/default.alpine.jsonl",
        generation_args="",
        judge_args="",
        judge_pipeline_args={},
        requires_sandbox=False,
        keep_mounts_for_sandbox=False,
        generation_module="nemo_skills.inference.eval.swe_bench",
        num_samples=0,
        num_chunks=None,
        eval_subfolder=f"eval-results/{benchmark}",
    )
    _patch_eval_for_sbatch_tests(monkeypatch, lambda: args)
    unit = eval_utils.EvalGenerationUnit(
        output_dir=str(tmp_path / args.eval_subfolder),
        input_file=args.input_file,
        extra_arguments="",
        random_seed=None,
        chunk_id=None,
        num_chunks=None,
        script=args.generation_module,
        requirements=[],
        wandb_parameters=None,
        with_sandbox=False,
    )
    args.job_ids = [0]
    monkeypatch.setattr(
        eval_pipeline,
        "prepare_eval_commands",
        lambda **kwargs: ({benchmark: args}, [([unit], {benchmark}, False, False, [])]),
    )
    submitted_pipelines = []
    monkeypatch.setattr(
        eval_pipeline.Pipeline,
        "run",
        lambda self, **kwargs: submitted_pipelines.append(self) or ["generation-0"],
    )

    eval_pipeline.eval(
        ctx=SimpleNamespace(args=[]),
        cluster="test-cluster",
        output_dir=str(tmp_path),
        benchmarks=benchmark,
        split="default.alpine",
        main_container="legacy-alpine-container",
        model="model",
        server_type="openai",
        server_address="http://server",
        auto_summarize_results=False,
        skip_hf_home_check=True,
    )

    assert submitted_pipelines[0].jobs[0]["group"].commands[-1].container == "legacy-alpine-container"


def test_eval_rejects_multiple_splits_for_other_benchmarks(monkeypatch, tmp_path):
    def benchmark_args():
        return eval_utils.BenchmarkArgs(
            name="gsm8k",
            input_file="/data/gsm8k/test.jsonl",
            generation_args="",
            judge_args="",
            judge_pipeline_args={},
            requires_sandbox=False,
            keep_mounts_for_sandbox=False,
            generation_module="nemo_skills.inference.generate",
            num_samples=0,
            num_chunks=None,
            eval_subfolder="eval-results/gsm8k",
        )

    _patch_eval_for_sbatch_tests(monkeypatch, benchmark_args)

    with pytest.raises(ValueError, match="only when evaluating SWE-bench Pro V1 or V2"):
        eval_pipeline.eval(
            ctx=SimpleNamespace(args=[]),
            cluster="test-cluster",
            output_dir=str(tmp_path),
            benchmarks="gsm8k",
            split="test,validation",
            model="model",
            server_type="openai",
            server_address="http://server",
        )


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
