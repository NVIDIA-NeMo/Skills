# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
NVEmbed judge evaluation script for MMAU-Pro closed-form questions.

This script handles:
1. Installing required packages (datasets, einops, transformers)
2. Copying generation output files to judge output directory
3. Running NVEmbed similarity matching evaluation
4. Creating .done markers for completed evaluations
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOG = logging.getLogger(__name__)


def install_packages():
    """Install required packages for NVEmbed evaluation."""
    LOG.info("Installing required packages...")
    subprocess.run(
        ["pip", "install", "-q", "datasets", "einops", "transformers==4.42.4", "tqdm"],
        check=True,
        capture_output=True,
        text=True,
    )
    LOG.info("Packages installed successfully")

    # Verify PyTorch and CUDA availability
    import torch

    LOG.info(
        f"PyTorch {torch.__version__} with CUDA {torch.version.cuda}, CUDA available: {torch.cuda.is_available()}"
    )


def load_nvembed_model(model_name: str = "nvidia/NV-Embed-v2"):
    """Load NVEmbed model using HuggingFace AutoModel with GPU support."""
    import torch
    from transformers import AutoModel

    if not hasattr(load_nvembed_model, "_cache"):
        load_nvembed_model._cache = {}

    if model_name in load_nvembed_model._cache:
        return load_nvembed_model._cache[model_name]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use explicit cache directory (respects HF_HOME env var)
    cache_dir = os.environ.get("HF_HOME")
    if cache_dir:
        LOG.info(f"Using HuggingFace cache directory: {cache_dir}")

    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, cache_dir=cache_dir, local_files_only=False)
    model.to(device)
    model.eval()
    load_nvembed_model._cache[model_name] = model
    LOG.info(f"Successfully loaded {model_name} on {device}")
    return model


def evaluate_with_nvembed_similarity(
    model_prediction: str, choices: list, ground_truth: str, model_name: str = "nvidia/NV-Embed-v2"
) -> tuple[str, float]:
    """NVEmbed-based evaluation: match predictions to choices using embedding similarity."""
    import torch
    import torch.nn.functional as F

    model = load_nvembed_model(model_name)
    device = next(model.parameters()).device

    with torch.no_grad():
        prediction_embedding = model.encode([model_prediction], instruction="", max_length=4096, device=device)
        choice_embeddings = model.encode(choices, instruction="", max_length=4096, device=device)

    prediction_embedding = F.normalize(prediction_embedding, p=2, dim=1)
    choice_embeddings = F.normalize(choice_embeddings, p=2, dim=1)

    scores = (prediction_embedding @ choice_embeddings.T) * 100
    scores = scores.squeeze()

    if scores.dim() == 0:
        scores = scores.unsqueeze(0)

    best_choice_idx = torch.argmax(scores).item()
    matched_choice = choices[best_choice_idx]
    confidence = torch.max(scores).item()

    return matched_choice, confidence


def evaluate_sample_with_nvembed(sample: dict[str, Any], model_name: str = "nvidia/NV-Embed-v2") -> dict[str, Any]:
    """Evaluate a single sample using NVEmbed similarity matching."""
    sample = sample.copy()

    if "nvembed_confidence" in sample:
        return sample

    generation = sample.get("generation", "").strip()
    choices = sample.get("choices", [])
    expected_answer = sample.get("expected_answer", "")

    # Fail fast if data is malformed - this indicates a pipeline error
    if not generation:
        raise ValueError(
            f"Sample missing generation field or has empty generation. Sample ID: {sample.get('id', 'unknown')}"
        )

    if not choices:
        raise ValueError(
            f"Sample missing choices field or has empty choices. Sample ID: {sample.get('id', 'unknown')}"
        )

    if not expected_answer:
        raise ValueError(f"Sample missing expected_answer field. Sample ID: {sample.get('id', 'unknown')}")

    matched_choice, confidence = evaluate_with_nvembed_similarity(generation, choices, expected_answer, model_name)
    is_correct = matched_choice.strip().lower() == expected_answer.strip().lower()

    sample.update(
        {"nvembed_matched_choice": matched_choice, "nvembed_confidence": confidence, "is_correct": is_correct}
    )
    return sample


def process_file(input_file: Path, output_file: Path, model_name: str = "nvidia/NV-Embed-v2"):
    """Copy input file to output location and run NVEmbed evaluation."""
    LOG.info(f"Processing {input_file} -> {output_file}")

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Copy input file to output location
    shutil.copy(input_file, output_file)
    LOG.info(f"Copied {input_file} to {output_file}")

    # Load data
    with open(output_file, "rt", encoding="utf-8") as fin:
        data = [json.loads(line) for line in fin]

    if not data:
        raise ValueError(f"Input file {input_file} is empty or contains no valid JSON lines")

    # Count samples to evaluate
    samples_to_evaluate = sum(1 for sample in data if "nvembed_confidence" not in sample)
    samples_already_done = len(data) - samples_to_evaluate

    if samples_already_done > 0:
        LOG.info(f"Resuming evaluation: {samples_already_done}/{len(data)} samples already have nvembed_confidence")

    # Evaluate each sample
    LOG.info(f"Evaluating {samples_to_evaluate} samples with NVEmbed")
    for idx, sample in enumerate(tqdm(data, desc="Evaluating samples")):
        data[idx] = evaluate_sample_with_nvembed(sample, model_name)

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
    parser = argparse.ArgumentParser(description="Run NVEmbed judge evaluation")
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
        "--num-seeds",
        type=int,
        default=1,
        help="Number of random seeds (for multiple seeds mode)",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip package installation (if already installed)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip processing if output files and .done markers already exist",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="nvidia/NV-Embed-v2",
        help="NVEmbed model to use for evaluation",
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

    # Check if all output files and .done markers already exist
    if args.skip_existing:
        all_done = True
        for _, output_file in files_to_process:
            done_file = Path(str(output_file) + ".done")
            if not (output_file.exists() and done_file.exists()):
                all_done = False
                break

        if all_done:
            LOG.info("All output files and .done markers already exist - skipping evaluation")
            return

    # Install packages unless skipped
    if not args.skip_install:
        install_packages()

    # Process all files
    LOG.info(f"Processing {len(files_to_process)} file(s)")
    for input_file, output_file in files_to_process:
        process_file(input_file, output_file, args.embedding_model)

    LOG.info("All files processed successfully")


if __name__ == "__main__":
    main()
