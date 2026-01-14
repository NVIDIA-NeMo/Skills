# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
Comet judge evaluation script for computing xCOMET-XXL machine translation metrics.

This script handles:
1. Copying generation output files to judge output directory
2. Running xCOMET-XXL model for machine translation evaluation
3. Creating .done markers for completed evaluations
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOG = logging.getLogger(__name__)


def load_comet_model(model_path: str):
    """Load xCOMET-XXL model with GPU support."""
    from comet import load_from_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_from_checkpoint(model_path)
    model.to(device)
    model.eval()
    LOG.info(f"Successfully loaded {model_path} on {device}")
    return model


def process_file(input_file: Path, output_file: Path, comet_model, batch_size: int = 16):
    """Copy input file to output location and run xCOMET-XXL evaluation."""
    LOG.info(f"Processing {input_file} -> {output_file}")

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Copy input file to output location
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    shutil.copy(input_file, output_file)
    LOG.info(f"Copied {input_file} to {output_file}")

    # Load data
    with open(output_file, "rt", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]

    if not data:
        raise ValueError(f"Input file {input_file} is empty or contains no valid JSON lines")

    comet_list = []
    for sample in data:
        try:
            comet_dict = {"src": sample["text"], "mt": sample["generation"], "gt": sample["translation"]}
            comet_list.append(comet_dict)
        except KeyError as e:
            LOG.error(f"Sample missing required field {e}: {sample}")
            raise ValueError(f"Sample missing required field: {e}")

    comet_scores = comet_model.predict(comet_list, batch_size=batch_size).scores

    for idx, sample in enumerate(data):
        data[idx]["comet"] = comet_scores[idx]

    # Write results
    with open(output_file, "wt", encoding="utf-8") as fout:
        for sample in data:
            fout.write(json.dumps(sample) + "\n")

    LOG.info(f"Evaluation completed for {output_file}")

    # Create .done marker
    done_file = Path(str(output_file) + ".done")
    done_file.touch()
    LOG.info(f"Created done marker: {done_file}")


def main():
    parser = argparse.ArgumentParser(description="Run xCOMET-XXL evaluation")
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to single input file (for single file mode)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Path to input directory (for multiple seeds mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to output directory",
    )
    parser.add_argument(
        "--comet-model-path",
        type=str,
        required=True,
        help="Path to xCOMET-XXL model to use for evaluation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for xCOMET-XXL inference",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of random seeds (for multiple seeds mode)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Determine which files to process
    files_to_process = []
    if args.input_file:
        # Single file mode
        input_file = Path(args.input_file)
        output_file = output_dir / "output.jsonl"
        files_to_process.append((input_file, output_file))
    elif args.input_dir:
        # Multiple seeds mode
        input_dir = Path(args.input_dir)
        for seed in range(args.num_seeds):
            input_file = input_dir / f"output-rs{seed}.jsonl"
            output_file = output_dir / f"output-rs{seed}.jsonl"
            files_to_process.append((input_file, output_file))
    else:
        LOG.error("Either --input-file or --input-dir must be specified")
        sys.exit(1)

    comet_model = load_comet_model(args.comet_model_path)
    # Process all files
    LOG.info(f"Processing {len(files_to_process)} file(s)")
    for input_file, output_file in files_to_process:
        process_file(input_file, output_file, comet_model, args.batch_size)

    LOG.info("All files processed successfully")


if __name__ == "__main__":
    main()
