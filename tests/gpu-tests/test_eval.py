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
import subprocess
from importlib import import_module
from pathlib import Path

import pytest
from utils import require_env_var

from nemo_skills.pipeline.cli import eval, prepare_data, run_cmd, wrap_arguments
from tests.conftest import docker_rm

# Datasets excluded from test_aaa_prepare_and_eval_all_datasets
# These don't support max_samples, require explicit parameters, or are very heavy to prepare
EXCLUDED_DATASETS = {
    "__pycache__",
    "ruler",
    "ruler2",
    "bigcodebench",
    "livecodebench",
    "livebench_coding",
    "livecodebench-pro",
    "livecodebench-cpp",
    "ioi",
    "bfcl_v3",
    "bfcl_v4",
    "swe-bench",
    "swe-bench-multilingual",
    "swe-rebench",
    "aai",
    "human-eval",
    "human-eval-infilling",
    "mbpp",
    "mmau-pro",
    "asr-leaderboard",
    "mrcr",
    "audiobench",
    "librispeech-pc",
    "musan",
    # Excluded for the time being as compute eval requires either a CTK or local docker engine to run
    "compute-eval",
}


def get_preparable_datasets():
    """Get list of datasets that can be prepared for testing."""
    datasets_dir = Path(__file__).absolute().parents[2] / "nemo_skills" / "dataset"
    return sorted(
        dataset.name
        for dataset in datasets_dir.iterdir()
        if dataset.is_dir() and (dataset / "prepare.py").exists() and dataset.name not in EXCLUDED_DATASETS
    )


# Run this test first to catch data prep failures early.
@pytest.mark.gpu
def test_aaa_prepare_and_eval_all_datasets():
    """Prepare and evaluate all datasets. Runs first to catch data prep failures early."""
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")
    model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")

    config_dir = Path(__file__).absolute().parent
    dataset_names = get_preparable_datasets()

    assert dataset_names, "No datasets found to prepare and evaluate"

    judge_datasets = []
    for dataset in dataset_names:
        dataset_module = import_module(f"nemo_skills.dataset.{dataset}")
        # Check if JUDGE_PIPELINE_ARGS exists (even if empty dict, which is falsy)
        if hasattr(dataset_module, "JUDGE_PIPELINE_ARGS"):
            judge_datasets.append(dataset)

    non_judge_datasets = [dataset for dataset in dataset_names if dataset not in judge_datasets]

    data_dir = Path(f"/tmp/nemo-skills-tests/{model_type}/data")
    docker_rm([str(data_dir)])

    # Prepare all datasets - fail fast if any dataset preparation fails
    exp = prepare_data(
        ctx=wrap_arguments(" ".join(dataset_names)),
        cluster="test-local",
        config_dir=str(config_dir),
        data_dir=str(data_dir),
        expname=f"prepare-all-datasets-{model_type}",
    )

    # Check experiment status - nemo_run doesn't raise exceptions on task failure
    status_dict = exp.status(return_dict=True)
    failed_tasks = []
    for task_name, status_info in status_dict.items():
        status = str(status_info.get("status", "")).upper()
        if "FAILED" in status or "ERROR" in status:
            failed_tasks.append(task_name)
    assert not failed_tasks, f"Data preparation tasks failed: {failed_tasks}"

    # Verify that data files were actually created for each dataset
    missing_datasets = []
    for dataset in dataset_names:
        dataset_path = data_dir / dataset
        if not dataset_path.exists():
            missing_datasets.append(dataset)
        else:
            # Check that at least one .jsonl file exists
            jsonl_files = list(dataset_path.glob("*.jsonl"))
            if not jsonl_files:
                missing_datasets.append(f"{dataset} (no .jsonl files)")

    assert not missing_datasets, f"Data files missing for datasets: {missing_datasets}"

    # Now run evaluation
    eval_kwargs = dict(
        cluster="test-local",
        config_dir=str(config_dir),
        data_dir=str(data_dir),
        model=model_path,
        server_type="sglang",
        server_gpus=1,
        server_nodes=1,
        auto_summarize_results=False,
    )

    common_ctx = "++max_samples=2 ++inference.tokens_to_generate=100 ++server.enable_soft_fail=True "

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/all-datasets-eval"
    docker_rm([output_dir])
    eval(
        ctx=wrap_arguments(common_ctx),
        output_dir=output_dir,
        benchmarks=",".join(non_judge_datasets),
        expname=f"eval-all-datasets-{model_type}",
        **eval_kwargs,
    )

    run_cmd(
        ctx=wrap_arguments(f"python -m nemo_skills.pipeline.summarize_results {output_dir}"),
        cluster="test-local",
        config_dir=str(config_dir),
    )

    eval_results_dir = Path(output_dir) / "eval-results"
    metrics_path = eval_results_dir / "metrics.json"
    assert metrics_path.exists(), "Missing aggregated metrics file"
    with metrics_path.open() as f:
        metrics = json.load(f)

    for dataset in non_judge_datasets:
        assert dataset in metrics, f"Missing metrics for {dataset}"

    # TODO: add same for judge_datasets after generate supports num_jobs
    # (otherwise it starts judge every time and takes forever)


@pytest.mark.gpu
def test_trtllm_eval():
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")
    model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/trtllm-eval"
    docker_rm([output_dir])

    cmd = (
        f"ns eval "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type trtllm "
        f"    --output_dir {output_dir} "
        f"    --benchmarks gsm8k "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    --server_args='--backend pytorch' "
        f"    ++max_samples=10 "
    )
    subprocess.run(cmd, shell=True, check=True)

    with open(f"{output_dir}/eval-results/gsm8k/metrics.json", "r") as f:
        metrics = json.load(f)["gsm8k"]["pass@1"]

    # rough check, since exact accuracy varies depending on gpu type
    if model_type == "qwen":
        assert metrics["symbolic_correct"] >= 70

    assert metrics["num_entries"] == 10


@pytest.mark.gpu
@pytest.mark.parametrize("server_type", ["trtllm"])
def test_trtllm_code_execution_eval(server_type):
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")
    model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")

    # we are using the base prompt for Qwen to make it follow few-shots
    if model_type == "qwen":
        # tokenizer = "Qwen/Qwen3-1.7B"
        code_tags = "qwen"
    else:
        raise ValueError("Only qwen models are supported in this test")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/{server_type}-eval"
    docker_rm([output_dir])

    cmd = (
        f"ns eval "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type {server_type} "
        f"    --output_dir {output_dir} "
        f"    --benchmarks gsm8k "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    --with_sandbox "
        f"    ++stop_phrase='\\n\\n\\n\\n\\n\\n' "
        f"    --server_args='--backend pytorch' "
        f"    ++tokenizer=Qwen/Qwen3-1.7B-Base "
        f"    ++code_tags={code_tags} "
        f"    ++examples_type=gsm8k_text_with_code "
        f"    ++max_samples=20 "
        f"    ++code_execution=True "
        f"    ++inference.tokens_to_generate=2048 "
    )
    subprocess.run(cmd, shell=True, check=True)

    with open(f"{output_dir}/eval-results/gsm8k/metrics.json", "r") as f:
        metrics = json.load(f)["gsm8k"]["pass@1"]
    # rough check, since exact accuracy varies depending on gpu type
    if model_type == "qwen":
        assert metrics["symbolic_correct"] >= 70
    assert metrics["num_entries"] == 20


@pytest.mark.gpu
@pytest.mark.parametrize(
    "server_type,server_args",
    [
        ("vllm", "--enforce-eager --max-model-len 4096"),
        ("sglang", "--context-length 4096"),
        ("trtllm", "--backend pytorch"),
    ],
)
def test_hf_eval(server_type, server_args):
    # this test expects qwen3-1.7b to properly check accuracy
    # will run a bunch of benchmarks, but is still pretty fast
    # mmlu/ifeval will be cut to 164 samples to save time
    # could cut everything, but human-eval/mbpp don't work with partial gens
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")
    model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")

    if model_type != "qwen":
        raise ValueError(f"Only running this test for qwen models, got {model_type}")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/{server_type}-eval"
    docker_rm([output_dir])

    cmd = (
        f"ns eval "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type {server_type} "
        f"    --output_dir {output_dir} "
        f"    --benchmarks algebra222,human-eval,ifeval,mmlu "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    --num_jobs 1 "
        f"    --server_args='{server_args}' "
        f"    ++max_samples=164 "
        f"    ++inference.tokens_to_generate=2048 "
        f"    ++parse_reasoning=True "
    )
    subprocess.run(cmd, shell=True, check=True)

    # checking that summarize results works (just that there are no errors, but can inspect the output as well)
    subprocess.run(
        f"ns summarize_results {output_dir}",
        shell=True,
        check=True,
    )

    with open(f"{output_dir}/eval-results/algebra222/metrics.json", "r") as f:
        metrics = json.load(f)["algebra222"]["pass@1"]

    assert metrics["symbolic_correct"] >= 75
    assert metrics["num_entries"] == 164

    with open(f"{output_dir}/eval-results/human-eval/metrics.json", "r") as f:
        metrics = json.load(f)["human-eval"]["pass@1"]

    assert metrics["passing_base_tests"] >= 35
    assert metrics["passing_plus_tests"] >= 35
    assert metrics["num_entries"] == 164

    with open(f"{output_dir}/eval-results/ifeval/metrics.json", "r") as f:
        metrics = json.load(f)["ifeval"]["pass@1"]

    assert metrics["prompt_strict_accuracy"] >= 60
    assert metrics["instruction_strict_accuracy"] >= 60
    assert metrics["prompt_loose_accuracy"] >= 60
    assert metrics["instruction_loose_accuracy"] >= 60
    assert metrics["num_prompts"] == 164

    with open(f"{output_dir}/eval-results/mmlu/metrics.json", "r") as f:
        metrics = json.load(f)["mmlu"]["pass@1"]
    assert metrics["symbolic_correct"] >= 60
    assert metrics["num_entries"] == 164


@pytest.mark.gpu
def test_megatron_eval():
    try:
        model_path = require_env_var("NEMO_SKILLS_TEST_MEGATRON_MODEL")
        model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")
    except ValueError:
        pytest.skip("Define NEMO_SKILLS_TEST_MEGATRON_MODEL and NEMO_SKILLS_TEST_MODEL_TYPE to run this test")

    if model_type != "qwen":
        raise ValueError(f"Only running this test for qwen models, got {model_type}")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/megatron-eval"
    docker_rm([output_dir])

    cmd = (
        f"ns eval "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type megatron "
        f"    --output_dir {output_dir} "
        f"    --benchmarks gsm8k "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    ++max_samples=5 "
        f"    ++tokenizer=Qwen/Qwen3-1.7B "
        f"    --server_args='--tokenizer-model Qwen/Qwen3-1.7B --inference-max-requests=20' "
    )
    subprocess.run(cmd, shell=True, check=True)

    # running compute_metrics to check that results are expected
    with open(f"{output_dir}/eval-results/gsm8k/metrics.json", "r") as f:
        metrics = json.load(f)["gsm8k"]["pass@1"]
    # rough check, since exact accuracy varies depending on gpu type
    # TODO: something is broken in megatron inference here as this should be 50!
    assert metrics["symbolic_correct"] >= 40
    assert metrics["num_entries"] == 5
