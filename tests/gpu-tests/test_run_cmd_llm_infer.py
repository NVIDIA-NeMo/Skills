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

import os
import subprocess
from pathlib import Path

import pytest
from utils import require_env_var

from tests.conftest import docker_rm


@pytest.mark.gpu
def test_run_cmd_llm_infer():
    """
    Uses (if available) VLLM servers, then sends the same prompt
    with with openai python api to check if generation works.
    """
    model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")

    model_info = [
        ("vllm", os.getenv("NEMO_SKILLS_TEST_HF_MODEL")),
    ]

    for server_type, model_path in model_info:
        if not model_path:
            continue

        output_dir = f"/tmp/nemo-skills-tests/{model_type}/{server_type}-run-cmd"

        docker_rm([output_dir])

        command = (
            f"cd /nemo_run/code/tests/scripts/ && "
            f"mkdir -p {output_dir} && "
            f"python run_cmd_llm_infer_check.py > {output_dir}/output.txt"
        )

        cmd = (
            f"ns run_cmd "
            f"--cluster test-local --config_dir {Path(__file__).absolute().parent} "
            f"--model {model_path} "
            f"--server_type {server_type} "
            f"--server_gpus 1 "
            f"--server_nodes 1 "
            f"--server_args '--enforce-eager' "
            f"--command '{command}'"
        )

        subprocess.run(cmd, shell=True, check=True)

        jsonl_file = Path(output_dir) / "output.txt"

        with open(jsonl_file, "r") as f:
            outputs = f.read()

        assert len(outputs) > 0  # just check that output text is not zero.
