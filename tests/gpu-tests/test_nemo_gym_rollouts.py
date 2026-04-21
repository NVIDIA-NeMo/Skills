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

from pathlib import Path

import pytest
from utils import require_env_var

from nemo_skills.pipeline.cli import nemo_gym_rollouts, wrap_arguments


@pytest.mark.gpu
def test_nemo_gym_rollouts_dry_run():
    """Test that nemo_gym_rollouts pipeline constructs correctly in dry-run mode."""
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")

    result = nemo_gym_rollouts(
        ctx=wrap_arguments(
            "+agent_name=math_with_judge_simple_agent "
            "+num_samples_in_parallel=4 "
            "+responses_create_params.max_output_tokens=256 "
            "+responses_create_params.temperature=1.0 "
        ),
        cluster="test-local",
        config_dir=str(Path(__file__).absolute().parent),
        config_paths="resources_servers/math_with_judge/configs/math_with_judge.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml",
        input_file="resources_servers/math_with_judge/data/example.jsonl",
        output_dir="/tmp/nemo-skills-tests/nemo-gym-rollouts-dry-run",
        model=model_path,
        server_type="vllm",
        server_gpus=1,
        server_args="--enforce-eager",
        dry_run=True,
    )

    # dry_run returns the pipeline result without executing
    assert result is not None


@pytest.mark.gpu
def test_nemo_gym_rollouts_dry_run_with_seeds():
    """Test that nemo_gym_rollouts creates separate jobs for each seed in dry-run mode."""
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")

    result = nemo_gym_rollouts(
        ctx=wrap_arguments("+agent_name=math_with_judge_simple_agent +num_samples_in_parallel=4 "),
        cluster="test-local",
        config_dir=str(Path(__file__).absolute().parent),
        config_paths="resources_servers/math_with_judge/configs/math_with_judge.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml",
        input_file="resources_servers/math_with_judge/data/example.jsonl",
        output_dir="/tmp/nemo-skills-tests/nemo-gym-rollouts-dry-run-seeds",
        model=model_path,
        server_type="vllm",
        server_gpus=1,
        server_args="--enforce-eager",
        num_random_seeds=3,
        dry_run=True,
    )

    assert result is not None
