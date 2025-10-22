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
import subprocess
from pathlib import Path

import pytest
from utils import require_env_var

from tests.conftest import docker_rm


@pytest.mark.gpu
def test_trtllm_judge():
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")
    model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")

    input_dir = "/nemo_run/code/tests/data"
    output_dir = f"/tmp/nemo-skills-tests/{model_type}/judge/math"

    docker_rm([output_dir])

    total_samples = 10
    cmd = (
        f"ns generate "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    --model {model_path} "
        f"    --server_type trtllm "
        f"    --generation_type=math_judge "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    --input_dir={input_dir} "
        f"    --output_dir={output_dir} "
        f"    --num_random_seeds=1 "
        f"    --preprocess_cmd='cp {input_dir}/output-rs0.test {input_dir}/output-rs0.jsonl' "
        f"    --server_args='--backend pytorch --max_batch_size {total_samples}' "
        f"    ++max_samples={total_samples} "
        f"    ++skip_filled=False "
        f"    ++inference.tokens_to_generate=2048 "
    )
    subprocess.run(cmd, shell=True, check=True)

    output_file = f"{output_dir}/output-rs0.jsonl"

    # no evaluation by default - checking just the number of lines and that there is a "judgement" key
    with open(output_file) as fin:
        lines = fin.readlines()
    assert len(lines) == total_samples
    for line in lines:
        data = json.loads(line)
        assert "judgement" in data

    # Adding a summarization step to check that the results are formatted correctly
    cmd = (
        f"ns summarize_results "
        f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
        f"    {os.path.dirname(output_file)} "
    )
    subprocess.run(cmd, shell=True, check=True)
