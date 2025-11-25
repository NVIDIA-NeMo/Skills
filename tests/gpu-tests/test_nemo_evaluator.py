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

"""
Test nemo_evaluator pipeline with self-hosted vLLM server.

This test validates that the nemo_evaluator command works correctly with:
- Self-hosted vLLM server
- Local cluster configuration (test-local)
- Minimal benchmark configuration (ifeval with 5 samples)
- Basic pipeline orchestration

The test uses the same model as other GPU tests (Qwen/Qwen3-1.7B) for consistency.
"""

import subprocess
from pathlib import Path

import pytest
from utils import require_env_var

from tests.conftest import docker_rm


@pytest.mark.gpu
def test_nemo_evaluator_vllm():
    """
    Test nemo_evaluator pipeline with self-hosted vLLM server.

    Uses minimal config with limited samples for quick testing.
    Validates that the pipeline runs successfully and produces output.
    """
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")
    model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")

    if model_type != "qwen":
        raise ValueError(f"Only running this test for qwen models, got {model_type}")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/nemo-evaluator-vllm"
    docker_rm([output_dir])

    config_path = Path(__file__).absolute().parent.parent / "data" / "nemo_evaluator" / "example-gpu-test-config.yaml"

    expname = "nemo-evaluator-expname"
    task_name = "ifeval"
    cmd = (
        f"ns nemo_evaluator "
        f"    --cluster test-local "
        f"    --config_dir {Path(__file__).absolute().parent} "
        f"    --output_dir {output_dir} "
        f"    --expname {expname} "
        f"    --server_type vllm "
        f"    --server_model {model_path} "
        f"    --server_gpus 1 "
        f"    --server_nodes 1 "
        f"    --server_args='--enforce-eager --max-model-len 4096' "
        f"    --nemo_evaluator_config {config_path} "
        f"    ++evaluation.nemo_evaluator_config.config.params.limit_samples=5 "
        f"    ++evaluation.nemo_evaluator_config.config.params.temperature=0.6 "
    )

    subprocess.run(cmd, shell=True, check=True)

    # Task dir outputs exist
    # {output_dir} / {expname} / nemo-evaluator-results/ task_name
    per_task_eval_dir = Path(output_dir) / expname / "nemo-evaluator-results" / task_name
    assert (per_task_eval_dir / "results.yml").exists()
