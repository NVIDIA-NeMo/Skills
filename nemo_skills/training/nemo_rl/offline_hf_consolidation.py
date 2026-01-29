# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

# This script can be used to consolidate sharded HF safetensors checkpoints
# to the consolidated format.

# Example model directory structure:
# model/
# ├── shard-00001-model-00001-of-00001.safetensors
# └── shard-00002-model-00001-of-00001.safetensors
#  ...

# This script works on both single and multiple workers:
# Example usage on 2 GPUs:
# torchrun --nproc-per-node=2 tools/offline_hf_consolidation.py --model-name meta-llama/Llama-3.2-1B --input-dir checkpoints/epoch_0_step_19/model/ --output-dir checkpoints/epoch_0_step_19/model/consolidated/
#
# Example usage on 1 GPU:
# python tools/offline_hf_consolidation.py --model-name meta-llama/Llama-3.2-1B --input-dir checkpoints/epoch_0_step_19/model/ --output-dir checkpoints/epoch_0_step_19/model/consolidated/

# copied from https://github.com/NVIDIA-NeMo/Automodel/blob/main/tools/offline_hf_consolidation.py

import argparse
import json
import os
import shutil

import torch
import torch.distributed as dist
from nemo_automodel.components.checkpoint._backports.consolidate_hf_safetensors import (
    consolidate_safetensors_files_on_every_rank,
)
from nemo_automodel.components.distributed.init_utils import (
    get_rank_safe,
    get_world_size_safe,
    initialize_distributed,
)


def copy_metadata_files(input_dir, output_dir):
    """
    Copy the metadata files over from the input directory to the output directory.
    """
    for item_name in os.listdir(input_dir):
        if item_name == "fqn_to_file_index_mapping.json":
            continue  # this is saved by the consolidation step
        src_path = os.path.join(input_dir, item_name)
        dst_path = os.path.join(output_dir, item_name)
        shutil.move(src_path, dst_path)
    shutil.rmtree(input_dir, ignore_errors=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Consolidate sharded HF safetensors checkpoints into consolidated files, "
            "preserving original sharding layout where possible."
        )
    )

    parser.add_argument(
        "--model-name",
        "-m",
        required=True,
        help=(
            "Hugging Face repo id (e.g. meta-llama/Llama-3.2-1B) or absolute path to a HF snapshot directory. "
            "Used as reference to copy metadata and derive FQN->file index mapping."
        ),
    )
    parser.add_argument(
        "--input-dir",
        "-i",
        required=True,
        help="Directory containing sharded safetensors files to consolidate.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory where consolidated safetensors and metadata will be written.",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=5,
        help="Number of threads for writing consolidated data (default: 5).",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "nccl", "gloo"],
        default="auto",
        help="Distributed backend to initialize (default: auto).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    backend = args.backend
    if backend == "auto":
        backend = "nccl" if torch.cuda.device_count() > 0 else "gloo"
    initialize_distributed(backend)

    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.input_dir):
        raise FileNotFoundError("Could not locate the input directory. Pass an absolute path to the input directory.")

    hf_metadata_dir = os.path.join(args.input_dir, ".hf_metadata")

    if not os.path.exists(hf_metadata_dir) and not os.path.isdir(hf_metadata_dir):
        raise FileNotFoundError("Expected to find the .hf_metadata directory in the input directory.")

    with open(os.path.join(hf_metadata_dir, "fqn_to_file_index_mapping.json"), "r") as f:
        fqn_to_index_mapping = json.load(f)

    consolidate_safetensors_files_on_every_rank(
        args.input_dir,
        args.output_dir,
        fqn_to_index_mapping,
        num_threads=args.num_threads,
    )

    if get_world_size_safe() > 1:
        dist.barrier()

    if get_rank_safe() == 0:
        copy_metadata_files(hf_metadata_dir, args.output_dir)

    if get_world_size_safe() > 1:
        dist.barrier()


if __name__ == "__main__":
    main()
