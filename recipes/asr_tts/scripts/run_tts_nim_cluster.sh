#!/bin/bash
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

# Example script for TTS generation using Riva NIM on cluster
# This script uses test data from tests/slurm-tests/tts_nim/

# Check required positional arguments
if [ $# -lt 2 ]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 CLUSTER_NAME WORKSPACE_PATH [CONTAINER]"
    echo ""
    echo "Arguments:"
    echo "  CLUSTER_NAME    Name of the cluster to run on"
    echo "  WORKSPACE_PATH  Path to the workspace directory"
    echo "  CONTAINER       (Optional) Container image to use"
    exit 1
fi

CLUSTER=$1
WORKSPACE=$2
CONTAINER=${3:-nvcr.io/nvstaging/nim/magpie-tts-multilingual:1.3.0-34013444}
SERVER_ARGS="--nim-tags-selector batch_size=64 --nim-disable-model-download false"
OUTPUT_DIR=/workspace/tts_outputs  # Relative to the mounted workspace

# Test data from slurm tests (automatically available in /nemo_run/code/ on cluster)
INPUT_FILE=/nemo_run/code/tests/slurm-tests/tts_nim/tts.test

MOUNTS="$WORKSPACE:/workspace"

ns generate \
    --cluster "$CLUSTER" \
    --model tts_nim \
    --generation_module recipes.asr_tts.riva_generate \
    --input_file "$INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --server_type generic \
    --num_chunks 1 \
    --server_entrypoint " DISABLE_RIVA_REALTIME_SERVER=True python3 -m nemo_skills.inference.server.serve_riva_nim" \
    --mount_paths "$MOUNTS" \
    --server_gpus 1 \
    --installation_command "pip install nvidia-riva-client==2.21.1" \
    --server_container "$CONTAINER" \
    --server_args "$SERVER_ARGS" \
    ++generation_type=tts \
    ++tts_output_dir="$OUTPUT_DIR/audio_outputs" \
    ++voice='Magpie-Multilingual.EN-US.Mia'
