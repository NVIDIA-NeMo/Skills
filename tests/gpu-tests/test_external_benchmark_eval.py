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
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest
from utils import require_env_var

from nemo_skills.pipeline.cli import eval, prepare_data, wrap_arguments
from nemo_skills.pipeline.start_server import launch_server, stop_server
from nemo_skills.pipeline.utils.packager import EXTERNAL_REPOS
from tests.conftest import docker_rm, docker_run

FIXTURE_DIR = Path(__file__).absolute().parents[1] / "data" / "dummy_external_benchmark"


# there is a built-in wait for server, but it doesn't have a timeout,
# so waiting here explicitly to not block ci jobs forever in case of problems
def _wait_for_server(server_address, timeout=300, interval=5):
    """Poll the server's /models endpoint until it responds or timeout is reached."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"{server_address}/models", timeout=5)
            return
        except urllib.error.URLError:
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
@pytest.mark.parametrize(
    "run_location",
    ["nemo_skills_repo", "outside_non_git_repo", "external_benchmark_git_repo"],
)
@pytest.mark.parametrize("use_data_dir", [True, False])
@pytest.mark.parametrize("use_full_path", [True, False])
def test_external_benchmark_prepare_and_eval(run_location, use_data_dir, use_full_path, sglang_server):
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")
    model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")

    config_dir = Path(__file__).absolute().parent
    path_suffix = "full-path" if use_full_path else "map-name"
    data_suffix = "with-data-dir" if use_data_dir else "no-data-dir"
    location_suffix = run_location.replace("_", "-")
    base_dir = Path(
        f"/tmp/nemo-skills-tests/{model_type}/external-bench-{location_suffix}-{path_suffix}-{data_suffix}"
    )
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
        f"chown -R {os.getuid()}:{os.getgid()} {base_dir}"
    )

    # Add to sys.path so custom generation modules are importable at submission time
    # Normally this would be handled by doing pip install, but since we are already in a running python process
    # this is a bit complicated, so we just add to sys.path directly
    sys.path.insert(0, str(ext_repo_dir))

    benchmark_map_path = str(ext_repo_dir / "benchmark_map.json")
    simple_bench_path = str(ext_repo_dir / "my_benchmarks" / "dataset" / "my_simple_bench")
    word_count_path = str(ext_repo_dir / "my_benchmarks" / "dataset" / "word_count")
    outside_non_git_dir = base_dir / "outside-non-git-run-dir"
    outside_non_git_dir.mkdir(parents=True, exist_ok=True)

    if run_location == "nemo_skills_repo":
        run_cwd = Path(__file__).absolute().parents[2]
    elif run_location == "outside_non_git_repo":
        run_cwd = outside_non_git_dir
    elif run_location == "external_benchmark_git_repo":
        run_cwd = ext_repo_dir
    else:
        raise ValueError(f"Unsupported run_location={run_location}")

    docker_rm([str(data_dir), str(output_dir)])

    saved_env = os.environ.get("NEMO_SKILLS_EXTRA_BENCHMARK_MAP")
    saved_uncommitted_check = os.environ.get("NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK")
    original_cwd = Path.cwd()
    try:
        os.chdir(run_cwd)
        os.environ["NEMO_SKILLS_EXTRA_BENCHMARK_MAP"] = benchmark_map_path
        # CI prepares built-in datasets before running tests, leaving untracked .jsonl
        # files in the repo. Disable the uncommitted-changes check so packaging succeeds.
        os.environ["NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK"] = "1"

        data_dir_arg = str(data_dir) if use_data_dir else None
        if use_full_path:
            simple_bench_arg = simple_bench_path
            word_count_arg = word_count_path
        else:
            simple_bench_arg = "my_simple_bench"
            word_count_arg = "word_count"

        # --- Prepare both benchmarks ---
        prepare_data(
            ctx=wrap_arguments(f"{simple_bench_arg} {word_count_arg}"),
            cluster="test-local",
            config_dir=str(config_dir),
            data_dir=data_dir_arg,
            expname=f"prepare-ext-bench-{location_suffix}-{path_suffix}-{model_type}",
        )

        # Check my_simple_bench prepared (1 sample)
        if use_data_dir:
            simple_prepared = data_dir / "my_simple_bench" / "test.jsonl"
        else:
            simple_prepared = Path(simple_bench_path) / "test.jsonl"
        assert simple_prepared.exists(), f"Expected {simple_prepared} to exist after prepare_data"
        with open(simple_prepared, "r") as f:
            assert len(f.readlines()) == 1

        # Check word_count prepared (5 samples)
        if use_data_dir:
            wc_prepared = data_dir / "word_count" / "test.jsonl"
        else:
            wc_prepared = Path(word_count_path) / "test.jsonl"
        assert wc_prepared.exists(), f"Expected {wc_prepared} to exist after prepare_data"
        with open(wc_prepared, "r") as f:
            assert len(f.readlines()) == 5

        # --- Eval both benchmarks (using the shared pre-launched server) ---
        eval(
            ctx=wrap_arguments(""),
            output_dir=str(output_dir),
            benchmarks=f"{simple_bench_arg},{word_count_arg}",
            cluster="test-local",
            config_dir=str(config_dir),
            data_dir=data_dir_arg,
            model=model_path,
            server_type="sglang",
            server_address=sglang_server,
            expname=f"eval-ext-bench-{location_suffix}-{path_suffix}-{model_type}",
        )

        # Check output for both benchmarks
        for bench_name in ["my_simple_bench", "word_count"]:
            eval_results_dir = Path(output_dir) / "eval-results" / bench_name
            output_files = list(eval_results_dir.glob("output*.jsonl"))
            assert output_files, f"No output files found in {eval_results_dir}"

            metrics_file = eval_results_dir / "metrics.json"
            assert metrics_file.exists(), f"Missing metrics.json for {bench_name}"
            with open(metrics_file, "r") as f:
                metrics = json.load(f)
            assert bench_name in metrics

    finally:
        os.chdir(original_cwd)
        sys.path.remove(str(ext_repo_dir))
        EXTERNAL_REPOS.pop("my_benchmarks")
        if saved_env is not None:
            os.environ["NEMO_SKILLS_EXTRA_BENCHMARK_MAP"] = saved_env
        else:
            os.environ.pop("NEMO_SKILLS_EXTRA_BENCHMARK_MAP", None)
        if saved_uncommitted_check is not None:
            os.environ["NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK"] = saved_uncommitted_check
        else:
            os.environ.pop("NEMO_SKILLS_DISABLE_UNCOMMITTED_CHANGES_CHECK", None)
