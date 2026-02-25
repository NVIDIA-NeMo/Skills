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

"""GPU integration tests for vLLM audio generation with VLLMMultimodalModel."""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest
from utils import require_env_var


@pytest.mark.gpu
def test_vllm_audio_generation():
    """Integration test: Generate with vLLM server using audio input."""
    model_path = require_env_var("NEMO_SKILLS_TEST_HF_MODEL")
    model_type = require_env_var("NEMO_SKILLS_TEST_MODEL_TYPE")

    output_dir = f"/tmp/nemo-skills-tests/{model_type}/vllm-audio-generation"
    # Clean up output directory
    if Path(output_dir).exists():
        shutil.rmtree(output_dir)

    # Create test input file with audio
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        test_data = [
            {
                "problem": "Transcribe this audio",
                "audio": {"path": "/nemo_run/code/tests/slurm-tests/asr_nim/wavs/t2_16.wav"},
            },
            {
                "problem": "What is in this audio?",
                "audio": {"path": "/nemo_run/code/tests/slurm-tests/asr_nim/wavs/t3_16.wav"},
            },
        ]
        for item in test_data:
            f.write(json.dumps(item) + "\n")
        input_file = f.name

    try:
        cmd = (
            f"ns generate "
            f"    --cluster test-local --config_dir {Path(__file__).absolute().parent} "
            f"    --model {model_path} "
            f"    --output_dir {output_dir} "
            f"    --server_type vllm_multimodal "
            f"    --server_gpus 1 "
            f"    --server_nodes 1 "
            f"    --server_args '--enforce-eager' "
            f"    --input_file={input_file} "
            f"    ++prompt_format=openai "
            f"    ++skip_filled=False "
        )
        subprocess.run(cmd, shell=True, check=True)

        # Verify output exists and has audio-related generation
        with open(f"{output_dir}/output.jsonl") as fin:
            lines = fin.readlines()

        assert len(lines) == 2, "Should have 2 output lines"

        for line in lines:
            data = json.loads(line)
            assert "generation" in data, "Should have generation field"
            assert len(data["generation"]) > 0, "Generation should not be empty"
            # If model supports audio, generation should contain something
            print(f"Generated: {data['generation']}")

    finally:
        # Cleanup temp file
        Path(input_file).unlink(missing_ok=True)
