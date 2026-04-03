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

"""HuggingFace Open ASR Leaderboard evaluation for NeMo ASR models.

Runs the HF Open ASR Leaderboard benchmark (8 datasets, WER metric) on
NeMo ASR models using the unified server with the nemo_asr or salm backend.

Uses 1 GPU for the server (NeMo ASR model). The evaluation client runs
on CPU alongside it.

Example usage::

    # Evaluate parakeet-v3 (traditional ASR model -> nemo_asr backend)
    python recipes/asr/run_hf_leaderboard.py \\
        --model nvidia/parakeet-tdt-0.6b-v3 \\
        --cluster oci_iad \\
        --output_dir /lustre/.../parakeet-v3-asr-leaderboard

    # Evaluate canary-qwen-2.5b (SALM model -> salm backend)
    python recipes/asr/run_hf_leaderboard.py \\
        --model nvidia/canary-qwen-2.5b \\
        --backend salm \\
        --cluster oci_iad \\
        --output_dir /lustre/.../canary-qwen-asr-leaderboard
"""

import argparse

from nemo_skills.pipeline.cli import eval, wrap_arguments

DEFAULT_SERVER_CONTAINER = "nvcr.io/nvidia/nemo:25.11"
DEFAULT_INSTALLATION_COMMAND = "pip install -r requirements/audio.txt"


def main():
    parser = argparse.ArgumentParser(description="Run HF Open ASR Leaderboard evaluation on a NeMo ASR model")
    parser.add_argument("--model", required=True, help="NeMo ASR model name or path (e.g. nvidia/canary-qwen-2.5b)")
    parser.add_argument("--cluster", required=True, help="Cluster name (e.g. oci_iad)")
    parser.add_argument("--output_dir", required=True, help="Directory for evaluation outputs")
    parser.add_argument("--data_dir", default="/dataset", help="Root data directory (must contain asr-leaderboard/)")
    parser.add_argument("--server_container", default=DEFAULT_SERVER_CONTAINER, help="NeMo container image")
    parser.add_argument("--server_gpus", type=int, default=1, help="Number of GPUs for the ASR server")
    parser.add_argument(
        "--backend",
        default="nemo_asr",
        choices=["nemo_asr", "salm"],
        help="Server backend: nemo_asr for traditional ASR models (parakeet), salm for SALM models (canary-qwen)",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="NeMo ASR transcription batch size")
    parser.add_argument(
        "--num_chunks", type=int, default=None, help="Split dataset into N chunks for data parallelism"
    )
    parser.add_argument("--expname", default="asr-leaderboard", help="Experiment name")
    parser.add_argument("--partition", default=None, help="Slurm partition (e.g. interactive)")
    parser.add_argument("--config_dir", default=None, help="Directory containing cluster config YAMLs")
    parser.add_argument("--split", default=None, help="Dataset split to evaluate (default: test = all datasets)")

    args = parser.parse_args()

    eval(
        ctx=wrap_arguments(
            "++prompt_format=openai "
            "++prompt_config=null "
            "++enable_audio=true "
            "++server.server_type=vllm_multimodal "
            "++max_concurrent_requests=16 "
            "++inference.tokens_to_generate=256"
        ),
        cluster=args.cluster,
        output_dir=args.output_dir,
        benchmarks="asr-leaderboard",
        model=args.model,
        server_type="generic",
        server_gpus=args.server_gpus,
        server_entrypoint=(
            "MKL_SERVICE_FORCE_INTEL=1 MKL_THREADING_LAYER=GNU "
            "MKL_NUM_THREADS=1 VML_NUM_THREADS=1 "
            "python -m nemo_skills.inference.server.serve_unified"
        ),
        server_args=f"--backend {args.backend} --batch_size {args.batch_size}",
        server_container=args.server_container,
        num_chunks=args.num_chunks,
        data_dir=args.data_dir,
        config_dir=args.config_dir,
        partition=args.partition,
        split=args.split,
        installation_command=DEFAULT_INSTALLATION_COMMAND,
        expname=args.expname,
    )


if __name__ == "__main__":
    main()
