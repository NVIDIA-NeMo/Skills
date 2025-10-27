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

# copied from https://github.com/NVIDIA-NeMo/RL/blob/main/examples/converters/convert_dcp_to_hf.py
# and added logic to figure out max step automatically

import argparse
import os
import re

import yaml
from nemo_rl.utils.native_checkpoint import convert_dcp_to_hf


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert Torch DCP checkpoint to HF checkpoint")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml file in the checkpoint directory",
    )
    parser.add_argument("--dcp-ckpt-path", type=str, default=None, help="Path to DCP checkpoint")
    parser.add_argument("--hf-ckpt-path", type=str, required=True, help="Path to save HF checkpoint")
    parser.add_argument(
        "--training-folder", type=str, default=None, help="Path to training folder containing step_X subfolders"
    )
    parser.add_argument(
        "--step", type=int, default=None, help="Step number to use from training folder (overrides highest found)"
    )
    parser.add_argument(
        "--max-position-embeddings",
        type=int,
        default=None,
        help="Max position embeddings to use for conversion. If not specified, will be inferred from the model config.",
    )

    # Parse known args for the script
    args = parser.parse_args()

    return args


def find_max_step_folder(training_folder, step_override=None):
    """
    Find the step_X folder with the highest X (or use step_override if given).
    Returns the path to the selected step_X folder, or None if not found.
    """
    # Check if 'checkpoints' is a subfolder and use it if present
    checkpoints_path = os.path.join(training_folder, "checkpoints")
    if os.path.isdir(checkpoints_path):
        training_folder = checkpoints_path
    step_pattern = re.compile(r"step_(\d+)")
    steps = []
    for entry in os.listdir(training_folder):
        match = step_pattern.fullmatch(entry)
        if match:
            steps.append(int(match.group(1)))
    if not steps:
        return None
    if step_override is not None:
        if step_override in steps:
            chosen_step = step_override
        else:
            raise ValueError(f"Specified step {step_override} not found in {training_folder}")
    else:
        chosen_step = max(steps)
    return os.path.join(training_folder, f"step_{chosen_step}")


def main():
    """Main entry point."""
    args = parse_args()

    # Raise error if both dcp_ckpt_path and training_folder are specified
    if args.dcp_ckpt_path and args.training_folder:
        raise ValueError("Specify only one of --dcp-ckpt-path or --training-folder, not both.")

    # If training-folder is specified, determine dcp_ckpt_path and config path automatically
    if args.training_folder:
        if args.config:
            raise ValueError(
                "Do not specify --config when using --training-folder; config.yaml will be read from the step_X folder."
            )
        step_folder = find_max_step_folder(args.training_folder, args.step)
        if not step_folder:
            raise RuntimeError(f"No step_X folders found in {args.training_folder}")
        dcp_ckpt_path = os.path.join(step_folder, "policy", "weights")
        config_path = os.path.join(step_folder, "config.yaml")
    else:
        dcp_ckpt_path = args.dcp_ckpt_path
        config_path = args.config

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_name_or_path = config["policy"]["model_name"]
    # TODO: After the following PR gets merged:
    # https://github.com/NVIDIA/NeMo-RL/pull/148/files
    # tokenizer should be copied from policy/tokenizer/* instead of relying on the model name
    # We can expose a arg at the top level --tokenizer_path to plumb that through.
    # This is more stable than relying on the current NeMo-RL get_tokenizer() which can
    # change release to release.
    tokenizer_name_or_path = config["policy"]["model_name"]

    print(f"Converting checkpoint from {dcp_ckpt_path} to {args.hf_ckpt_path}")
    print(f"Using tokenizer from: {tokenizer_name_or_path}")

    hf_overrides = {}
    if args.max_position_embeddings:
        hf_overrides["max_position_embeddings"] = args.max_position_embeddings

    hf_ckpt = convert_dcp_to_hf(
        dcp_ckpt_path=dcp_ckpt_path,
        hf_ckpt_path=args.hf_ckpt_path,
        model_name_or_path=model_name_or_path,
        tokenizer_name_or_path=tokenizer_name_or_path,
        overwrite=True,
        hf_overrides=hf_overrides,
    )
    print(f"Saved HF checkpoint to: {hf_ckpt}")


if __name__ == "__main__":
    main()
