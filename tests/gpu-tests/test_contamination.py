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
import os
from pathlib import Path

import pytest

from nemo_skills.pipeline.cli import generate, run_cmd, wrap_arguments
from tests.conftest import docker_rm


@pytest.mark.gpu
def test_check_contamination():
    model_path = os.getenv("NEMO_SKILLS_TEST_HF_MODEL")
    if not model_path:
        pytest.skip("Define NEMO_SKILLS_TEST_HF_MODEL to run this test")
    model_type = os.getenv("NEMO_SKILLS_TEST_MODEL_TYPE")
    if not model_type:
        pytest.skip("Define NEMO_SKILLS_TEST_MODEL_TYPE to run this test")
    if model_type != "llama":
        pytest.skip("Only running this test for llama models")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/contamination"

    docker_rm([output_dir])

    test_sets = ["math-500", "amc23", "aime24"]
    retrieve_from = ",".join(f"/nemo_run/code/nemo_skills/dataset/{test_set}/test.jsonl" for test_set in test_sets)

    cmd = (
        f"python -m nemo_skills.inference.retrieve_similar "
        f"    ++retrieve_from=\\'{retrieve_from}\\' "
        f"    ++compare_to=/nemo_run/code/tests/data/contamination-example.test "
        f"    ++output_file='{output_dir}/math-contamination-retrieved.jsonl' "
        f"    ++top_k=1 "
    )

    run_cmd(
        cluster="test-local",
        config_dir=Path(__file__).absolute().parent,
        container="nemo-rl",
        num_gpus=1,
        ctx=wrap_arguments(cmd),
    )

    generate(
        ctx=wrap_arguments(""),
        cluster="test-local",
        config_dir=Path(__file__).absolute().parent,
        generation_type="check_contamination",
        input_file=f"{output_dir}/math-contamination-retrieved.jsonl",
        output_dir=output_dir,
        model=model_path,
        server_type="trtllm",
        server_args="--backend pytorch",
        server_gpus=1,
    )

    output_file = f"{output_dir}/output.jsonl"

    with open(output_file) as fin:
        lines = fin.readlines()
    assert len(lines) == 10
    num_contaminated = 0
    for line in lines:
        data = json.loads(line)
        assert "contaminated" in data
        num_contaminated += data["contaminated"]
    # gt answer is 4, but llama judges more problems as contaminated
    assert 3 <= num_contaminated < 10
