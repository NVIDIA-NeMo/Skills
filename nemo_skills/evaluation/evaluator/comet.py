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
import sys
from pathlib import Path

import torch
import torch.distributed as dist

# unbabel-comet's CustomWriter.gather_all_predictions() reloads the per-rank
# Prediction shards it just wrote with torch.save into a private tempdir.
# PyTorch >=2.6 flipped torch.load's default to weights_only=True, which
# refuses to unpickle COMET's `Prediction` (an OrderedDict subclass) and
# raises `_pickle.UnpicklingError: Weights only load failed ...
_orig_torch_load = torch.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_compat

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOG = logging.getLogger(__name__)


def _is_global_rank_zero() -> bool:
    """True if this process is global rank 0."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True


_PRECISION_TO_DTYPE: dict[str, torch.dtype | None] = {
    "fp32": None,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}


def load_comet_model(model_path: str, dtype: torch.dtype | None = None):
    """Load COMET on CPU, optionally with lower precision.

    Do not move it to GPU here: in multi-GPU DDP, each spawned rank would run this
    function, and `.to("cuda")` would put every copy on cuda:0, causing OOM.
    `Predictor.predict(gpus=N)` later moves each rank to the correct device.
    """
    from comet import load_from_checkpoint

    model = load_from_checkpoint(model_path)
    if dtype is not None:
        model = model.to(dtype=dtype)
        LOG.info(f"Cast COMET model weights to {dtype}")
    LOG.info(f"Loaded COMET checkpoint from {model_path} (device placement deferred to Trainer)")
    return model


def process_file(
    input_file: Path,
    output_file: Path,
    comet_model,
    batch_size: int = 16,
):
    """Score one inference output file with COMET."""
    LOG.info(f"Processing {input_file} -> {output_file}")

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file, "rt", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]

    if not data:
        raise ValueError(f"Input file {input_file} is empty or contains no valid JSON lines")

    comet_list = []
    for sample in data:
        try:
            comet_list.append(
                {
                    "src": sample["source"],
                    "mt": sample["generation"],
                    "ref": sample["reference"],
                }
            )
        except KeyError as e:
            LOG.error(f"Sample missing required field {e}: {sample}")
            raise ValueError(f"Sample missing required field: {e}")

    num_gpus = torch.cuda.device_count()
    LOG.info(f"Predicting COMET model with {num_gpus} GPUs")
    prediction = comet_model.predict(comet_list, batch_size=batch_size, gpus=num_gpus)

    if not _is_global_rank_zero():
        # Non-zero ranks have already done their share of the DDP work; skip I/O.
        return

    comet_scores = prediction.scores
    for idx, sample in enumerate(data):
        data[idx]["comet"] = comet_scores[idx]

    output_file.parent.mkdir(parents=True, exist_ok=True)
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
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=sorted(_PRECISION_TO_DTYPE.keys()),
        help=(
            "Numerical precision for COMET inference. 'bf16' is recommended on H100/A100 "
            "(~2x faster than fp32, half the per-rank weight memory, same dynamic range as fp32). "
            "Use 'fp32' to reproduce the original behavior."
        ),
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

    comet_model = load_comet_model(args.comet_model_path, dtype=_PRECISION_TO_DTYPE[args.precision])
    # Process all files
    LOG.info(f"Processing {len(files_to_process)} file(s)")
    for input_file, output_file in files_to_process:
        process_file(
            input_file,
            output_file,
            comet_model,
            args.batch_size,
        )

    LOG.info("All files processed successfully")


if __name__ == "__main__":
    main()
