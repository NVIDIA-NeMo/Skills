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
import time
import urllib.request
from pathlib import Path

import pytest
from utils import require_env_var

from nemo_skills.pipeline.cli import eval, prepare_data, wrap_arguments
from nemo_skills.pipeline.start_server import launch_server, stop_server
from tests.conftest import docker_rm, docker_run

FIXTURE_DIR = Path(__file__).absolute().parents[1] / "data" / "dummy_external_benchmark"


# there is a built-in wait for server, but it doesn't have a timeout,
# so waiting here explicitly to not block ci jobs forever in case of problems
def _wait_for_server(server_address, timeout=600, interval=5):
    """Poll the server's /models endpoint until it responds or timeout is reached."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"{server_address}/models", timeout=5)
            return
        except Exception:
            time.sleep(interval)
    raise RuntimeError(f"Server at {server_address} failed to start within {timeout}s")


# TODO: start using this in main test_eval.py as well and enable judge benchmarks through this
@pytest.fixture(scope="module")
def sglang_server():
    """Start a shared sglang server for all tests in this module."""
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")
    config_dir = str(Path(__file__).absolute().parent)

    exp, server_port = launch_server(
        cluster="test-local",
        config_dir=config_dir,
        model=model_path,
        server_type="sglang",
        server_gpus=1,
    )

    server_address = f"http://localhost:{server_port}/v1"
    try:
        _wait_for_server(server_address)
    except Exception:
        stop_server(exp)
        raise

    yield server_address

    stop_server(exp)


@pytest.mark.gpu
@pytest.mark.parametrize("use_data_dir", [True, False])
@pytest.mark.parametrize("use_full_path", [True, False])
def test_external_benchmark_prepare_and_eval(use_data_dir, use_full_path, sglang_server):
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")
    model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")

    config_dir = Path(__file__).absolute().parent
    path_suffix = "full-path" if use_full_path else "map-name"
    data_suffix = "with-data-dir" if use_data_dir else "no-data-dir"
    base_dir = Path(f"/tmp/nemo-skills-tests/{model_type}/external-bench-{path_suffix}-{data_suffix}")
    data_dir = base_dir / "data"
    output_dir = base_dir / "eval-output"

    # Copy fixture to /tmp so docker mounts work via /tmp:/tmp
    ext_repo_dir = base_dir / "dummy_external_benchmark"
    docker_rm([str(ext_repo_dir)])
    # mounting /tmp and also main repo folder to be able to copy things
    repo_path = Path(__file__).absolute().parents[2]
    docker_run(
        f"mkdir -p {ext_repo_dir.parent} && cp -r {FIXTURE_DIR} {ext_repo_dir}",
        volume_paths=["/tmp:/tmp", f"{repo_path}:{repo_path}"],
    )

    # Init git (needed for container packaging) and fix ownership so host user can access
    docker_run(
        f"apk add --no-cache git && cd {ext_repo_dir} && git init && git add . && "
        f"GIT_AUTHOR_NAME=test GIT_AUTHOR_EMAIL=t@t GIT_COMMITTER_NAME=test GIT_COMMITTER_EMAIL=t@t "
        f"git commit -m init --no-gpg-sign && "
        f"chown -R {os.getuid()}:{os.getgid()} {ext_repo_dir}"
    )

    benchmark_map_path = str(ext_repo_dir / "benchmark_map.json")
    simple_bench_path = str(ext_repo_dir / "my_benchmarks" / "dataset" / "my_simple_bench")

    docker_rm([str(data_dir), str(output_dir)])

    saved_env = os.environ.get("NEMO_SKILLS_EXTRA_BENCHMARK_MAP")
    try:
        os.environ["NEMO_SKILLS_EXTRA_BENCHMARK_MAP"] = benchmark_map_path

        data_dir_arg = str(data_dir) if use_data_dir else None
        benchmark_arg = simple_bench_path if use_full_path else "my_simple_bench"

        # --- Prepare ---
        prepare_data(
            ctx=wrap_arguments(benchmark_arg),
            cluster="test-local",
            config_dir=str(config_dir),
            data_dir=data_dir_arg,
            expname=f"prepare-ext-bench-{path_suffix}-{model_type}",
        )

        if use_data_dir:
            prepared_file = data_dir / "my_simple_bench" / "test.jsonl"
        else:
            prepared_file = Path(simple_bench_path) / "test.jsonl"
        assert prepared_file.exists(), f"Expected {prepared_file} to exist after prepare_data"
        with open(prepared_file, "r") as f:
            lines = f.readlines()
        assert len(lines) == 1

        # --- Eval (using the shared pre-launched server) ---
        eval(
            ctx=wrap_arguments(""),
            output_dir=str(output_dir),
            benchmarks=benchmark_arg,
            cluster="test-local",
            config_dir=str(config_dir),
            data_dir=data_dir_arg,
            model=model_path,
            server_type="sglang",
            server_address=sglang_server,
            expname=f"eval-ext-bench-{path_suffix}-{model_type}",
        )

        # Check output
        eval_results_dir = Path(output_dir) / "eval-results" / "my_simple_bench"
        output_files = list(eval_results_dir.glob("output*.jsonl"))
        assert output_files, f"No output files found in {eval_results_dir}"

        # Check metrics.json
        metrics_file = Path(output_dir) / "eval-results" / "metrics.json"
        assert metrics_file.exists(), "Missing metrics.json"
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
        assert "my_simple_bench" in metrics

    finally:
        if saved_env is not None:
            os.environ["NEMO_SKILLS_EXTRA_BENCHMARK_MAP"] = saved_env
        else:
            os.environ.pop("NEMO_SKILLS_EXTRA_BENCHMARK_MAP", None)
