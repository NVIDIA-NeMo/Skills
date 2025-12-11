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

import subprocess
from types import SimpleNamespace

import pytest

from nemo_skills.pipeline.utils import get_mounted_path
from nemo_skills.pipeline.utils.eval import get_benchmark_args_from_module


def test_error_on_extra_params():
    """Testing that when we pass in any unsupported parameters, there is an error."""

    # top-level
    # test is not supported
    cmd = (
        "python nemo_skills/inference/generate.py "
        "    ++prompt_config=generic/math "
        "    ++output_file=./test-results/gsm8k/output.jsonl "
        "    ++input_file=./nemo_skills/dataset/gsm8k/test.jsonl "
        "    ++server.server_type=trtllm "
        "    ++test=1"
    )
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        assert "got an unexpected keyword argument 'test'" in e.stderr.decode()

    # inside nested dataclass
    cmd = (
        "python nemo_skills/inference/generate.py "
        "    ++prompt_config=generic/math "
        "    ++output_file=./test-results/gsm8k/output.jsonl "
        "    ++inference.num_few_shots=0 "
        "    ++input_file=./nemo_skills/dataset/gsm8k/test.jsonl "
        "    ++server.server_type=vllm "
    )
    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        assert "got an unexpected keyword argument 'num_few_shots'" in e.stderr.decode()


@pytest.mark.parametrize(
    "mount_source, mount_dest, input_path, expected",
    [
        # Original path should be mapped correctly
        ("/lustre/data", "/data", "/lustre/data/my_path.jsonl", "/data/my_path.jsonl"),
        ("/lustre/data/", "/data", "/lustre/data/my_path.jsonl", "/data/my_path.jsonl"),
        ("/lustre/data", "/data/", "/lustre/data/my_path.jsonl", "/data/my_path.jsonl"),
        ("/lustre/data/", "/data/", "/lustre/data/my_path.jsonl", "/data/my_path.jsonl"),
        # Already mounted path should return unchanged
        ("/lustre/data", "/data", "/data/my_path.jsonl", "/data/my_path.jsonl"),
        # Fallback mount - match broader /lustre if more specific one is not present
        ("/lustre", "/lustre", "/lustre/data/my_path.jsonl", "/lustre/data/my_path.jsonl"),
    ],
)
def test_get_mounted_path(mount_source, mount_dest, input_path, expected):
    """
    Test get_mounted_path with various combinations of mount source/destination paths
    and input paths, including trailing slashes and already-mounted paths.
    """
    cluster_config = {
        "mounts": [f"{mount_source}:{mount_dest}"],
        "executor": "slurm",
    }

    result = get_mounted_path(cluster_config, input_path)
    assert result == expected


def test_get_benchmark_args_input_file_should_be_local_path_for_executor_none(tmp_path):
    """For executor='none', input_file should be a real local path, not a container path."""
    # Setup: create a local data file
    benchmark_dir = tmp_path / "gsm8k"
    benchmark_dir.mkdir()
    (benchmark_dir / "test.jsonl").write_text('{"problem": "test"}\n')

    cluster_config = {"executor": "none", "containers": {}}
    mock_module = SimpleNamespace(
        EVAL_SPLIT="test",
        PROMPT_CONFIG="",
        GENERATION_ARGS="",
        EVAL_ARGS="",
        REQUIRES_SANDBOX=False,
        KEEP_MOUNTS_FOR_SANDBOX=False,
        GENERATION_MODULE="nemo_skills.inference.generate",
        JUDGE_PIPELINE_ARGS={},
        JUDGE_ARGS="",
        NUM_SAMPLES=0,
        NUM_CHUNKS=0,
    )

    result = get_benchmark_args_from_module(
        benchmark_module=mock_module,
        benchmark="gsm8k",
        split="test",
        cluster_config=cluster_config,
        data_path=str(tmp_path),  # local path like /tmp/pytest-xxx
        is_on_cluster=False,
        eval_requires_judge=False,
    )

    # For executor='none' (no container), input_file should be the actual local path
    expected_input_file = str(tmp_path / "gsm8k" / "test.jsonl")
    assert result.input_file == expected_input_file, (
        f"Expected local path {expected_input_file}, got {result.input_file}"
    )
