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

"""
MMMU-Pro dataset preparation script.

Uses the 'vision' config which has augmented settings where images are critical.

Usage:
    ns prepare_data mmmu-pro --data_dir <path_to_data_dir>
"""

import argparse
import ast
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from nemo_skills.dataset.utils import get_mcq_fields


def format_entry(entry, images_dir: Path) -> dict | None:
    """Format a MMMU-Pro entry for NeMo-Skills VLM evaluation."""
    # Skip entries without images
    if entry["image"] is None:
        return None

    image_filename = f"{entry['id']}.png"
    image_path = images_dir / image_filename
    entry["image"].save(image_path)

    # Parse options - it's a string representation of a list
    options = ast.literal_eval(entry["options"])

    # Build MCQ fields using standard utility
    mcq_fields = get_mcq_fields("", options)  # Question is in the image for 'vision' config

    # Subject for subset metrics
    subject = entry["subject"].replace(" ", "_")

    return {
        "problem": mcq_fields["problem"],
        "image_path": f"images/{image_filename}",
        "expected_answer": entry["answer"],
        "subset_for_metrics": subject,
        "id": entry["id"],
    }


def save_data(split: str, data_dir: Path):
    """Download and prepare MMMU-Pro data."""
    images_dir = data_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    output_file = data_dir / f"{split}.jsonl"

    print(f"Loading MMMU-Pro vision {split} split...")
    dataset = load_dataset("MMMU/MMMU_Pro", "vision", split=split)

    print(f"Processing {len(dataset)} entries...")
    count = 0
    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in tqdm(dataset, desc=f"Preparing {split}"):
            formatted = format_entry(entry, images_dir)
            if formatted is None:
                continue
            fout.write(json.dumps(formatted) + "\n")
            count += 1

    print(f"✓ Saved {count} entries to {output_file}")
    print(f"✓ Images saved to {images_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare MMMU-Pro dataset")
    parser.add_argument(
        "--split",
        default="test",
        choices=("test",),
        help="Dataset split to prepare (default: test)",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.getenv("NEMO_SKILLS_DATA_DIR"),
        help="Directory to save prepared data. Defaults to NEMO_SKILLS_DATA_DIR env var.",
    )
    args = parser.parse_args()

    # Resolve data_dir
    if args.data_dir:
        data_dir = Path(args.data_dir) / "mmmu-pro"
    else:
        pkg_dir = Path(__file__).absolute().parent
        print(
            "Warning: --data_dir not specified and NEMO_SKILLS_DATA_DIR not set. "
            f"Writing to package directory: {pkg_dir}",
            file=sys.stderr,
        )
        data_dir = pkg_dir

    save_data(args.split, data_dir)
