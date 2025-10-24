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

import pytest

from nemo_skills.pipeline.utils import get_mounted_path


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
