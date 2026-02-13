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
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from utils import require_env_var

from nemo_skills.pipeline.cli import eval, prepare_data, run_cmd, wrap_arguments
from tests.conftest import docker_rm

FIXTURE_DIR = Path(__file__).absolute().parents[1] / "data" / "dummy_external_benchmark"


@pytest.mark.gpu
def test_external_benchmark_prepare_and_eval():
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")
    model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")

    config_dir = Path(__file__).absolute().parent
    base_dir = Path(f"/tmp/nemo-skills-tests/{model_type}/external-bench")
    data_dir = base_dir / "data"
    output_dir = base_dir / "eval-output"

    # Copy fixture to /tmp so docker mounts work via /tmp:/tmp
    ext_repo_dir = base_dir / "dummy_external_benchmark"
    if ext_repo_dir.exists():
        docker_rm([str(ext_repo_dir)])
    shutil.copytree(FIXTURE_DIR, ext_repo_dir)

    # Init git (needed for container packaging)
    git_env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "test",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "test",
        "GIT_COMMITTER_EMAIL": "t@t",
    }
    subprocess.run(["git", "init"], cwd=ext_repo_dir, check=True)
    subprocess.run(["git", "add", "."], cwd=ext_repo_dir, check=True)
    subprocess.run(["git", "commit", "-m", "init", "--no-gpg-sign"], cwd=ext_repo_dir, check=True, env=git_env)

    benchmark_map_path = str(ext_repo_dir / "benchmark_map.json")
    simple_bench_path = str(ext_repo_dir / "my_benchmarks" / "dataset" / "my_simple_bench")

    docker_rm([str(data_dir), str(output_dir)])

    saved_env = os.environ.get("NEMO_SKILLS_EXTRA_BENCHMARK_MAP")
    try:
        # --- Test via map name ---
        os.environ["NEMO_SKILLS_EXTRA_BENCHMARK_MAP"] = benchmark_map_path

        # Prepare data
        prepare_data(
            ctx=wrap_arguments("my_simple_bench"),
            cluster="test-local",
            config_dir=str(config_dir),
            data_dir=str(data_dir),
            expname=f"prepare-ext-bench-{model_type}",
        )

        # Verify test.jsonl was created
        prepared_file = data_dir / "my_simple_bench" / "test.jsonl"
        assert prepared_file.exists(), f"Expected {prepared_file} to exist after prepare_data"
        with open(prepared_file, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1

        # Evaluate
        eval(
            ctx=wrap_arguments("++max_samples=1 ++inference.tokens_to_generate=100 ++server.enable_soft_fail=True"),
            output_dir=str(output_dir),
            benchmarks="my_simple_bench",
            cluster="test-local",
            config_dir=str(config_dir),
            data_dir=str(data_dir),
            model=model_path,
            server_type="sglang",
            server_gpus=1,
            server_nodes=1,
            expname=f"eval-ext-bench-{model_type}",
            auto_summarize_results=False,
        )

        # Check output.jsonl
        eval_results_dir = Path(output_dir) / "eval-results" / "my_simple_bench"
        output_files = list(eval_results_dir.glob("output*.jsonl"))
        assert output_files, f"No output files found in {eval_results_dir}"

        # Summarize results
        run_cmd(
            ctx=wrap_arguments(f"python -m nemo_skills.pipeline.summarize_results {output_dir}"),
            cluster="test-local",
            config_dir=str(config_dir),
        )

        # Check metrics.json
        metrics_file = eval_results_dir / "metrics.json"
        assert metrics_file.exists(), "Missing metrics.json"
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        assert "my_simple_bench" in metrics

        # --- Test via full path ---
        output_dir_path = base_dir / "eval-output-path"
        docker_rm([str(output_dir_path)])

        eval(
            ctx=wrap_arguments("++max_samples=1 ++inference.tokens_to_generate=100 ++server.enable_soft_fail=True"),
            output_dir=str(output_dir_path),
            benchmarks=simple_bench_path,
            cluster="test-local",
            config_dir=str(config_dir),
            data_dir=str(data_dir),
            model=model_path,
            server_type="sglang",
            server_gpus=1,
            server_nodes=1,
            expname=f"eval-ext-bench-path-{model_type}",
            auto_summarize_results=True,
        )

        path_eval_results = Path(output_dir_path) / "eval-results" / "my_simple_bench"
        path_output_files = list(path_eval_results.glob("output*.jsonl"))
        assert path_output_files, f"No output files found for full path eval in {path_eval_results}"

    finally:
        if saved_env is not None:
            os.environ["NEMO_SKILLS_EXTRA_BENCHMARK_MAP"] = saved_env
        else:
            os.environ.pop("NEMO_SKILLS_EXTRA_BENCHMARK_MAP", None)
