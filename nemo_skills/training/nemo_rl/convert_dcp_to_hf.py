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
import json
import os
import re
import shutil
import subprocess

import yaml


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


def is_safetensors_checkpoint(weights_path):
    """Check if checkpoint is in the new safetensors format (has model/.hf_metadata/)."""
    hf_metadata_path = os.path.join(weights_path, "model", ".hf_metadata")
    return os.path.isdir(hf_metadata_path)


def copy_tokenizer_files(tokenizer_path, hf_ckpt_path):
    """Copy tokenizer files from the original model to the HF checkpoint directory.

    Args:
        tokenizer_path: Path to directory containing tokenizer files
        hf_ckpt_path: Path to the HF checkpoint directory
    """
    tokenizer_files = [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
        "added_tokens.json",
        "chat_template.jinja",
    ]
    for fname in tokenizer_files:
        src = os.path.join(tokenizer_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(hf_ckpt_path, fname))
            print(f"Copied {fname}")


def convert_safetensors_to_hf(weights_path, hf_ckpt_path, model_name, tokenizer_path, hf_overrides=None):
    """Convert safetensors checkpoint to HF format using offline_hf_consolidation.py."""
    model_dir = os.path.join(weights_path, "model")

    # Get the path to the consolidation script (same directory as this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    consolidation_script = os.path.join(script_dir, "offline_hf_consolidation.py")

    # Run the consolidation script using uv with the automodel extra to get nemo_automodel
    # Reference: https://github.com/NVIDIA-NeMo/Automodel/blob/main/tools/offline_hf_consolidation.py
    cmd = [
        "uv",
        "run",
        "--active",
        "--extra",
        "automodel",
        "python",
        consolidation_script,
        "--model-name",
        model_name,
        "--input-dir",
        model_dir,
        "--output-dir",
        hf_ckpt_path,
    ]

    print(f"Running consolidation: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Copy tokenizer files (not handled by offline consolidation)
    # TODO: this will fail if config["policy"]["model_name"] isn't a path, but that's not common and we should
    # anyway remove this logic when it's properly handled in nemo-rl
    copy_tokenizer_files(tokenizer_path, hf_ckpt_path)

    # Apply hf_overrides to config.json if provided
    if hf_overrides:
        config_path = os.path.join(hf_ckpt_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        config.update(hf_overrides)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

    return hf_ckpt_path


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

    # Check if checkpoint is in the new safetensors format
    if is_safetensors_checkpoint(dcp_ckpt_path):
        print("Detected safetensors checkpoint format, using offline consolidation...")
        hf_ckpt = convert_safetensors_to_hf(
            weights_path=dcp_ckpt_path,
            hf_ckpt_path=args.hf_ckpt_path,
            model_name=model_name_or_path,
            tokenizer_path=tokenizer_name_or_path,
            hf_overrides=hf_overrides if hf_overrides else None,
        )
    else:
        print("Detected DCP checkpoint format, using DCP conversion...")
        from nemo_rl.utils.native_checkpoint import convert_dcp_to_hf

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
