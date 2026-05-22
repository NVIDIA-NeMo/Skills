#!/usr/bin/env python3
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

import argparse
import json
from pathlib import Path
from typing import Dict, List


def collect_datasets(root_dir: Path):
    """Collect text datasets from numeric subfolders and remap folder ids sequentially."""
    if not root_dir.exists() or not root_dir.is_dir():
        raise FileNotFoundError(f"Root directory not found or not a directory: {root_dir}")

    data: Dict[str, List[dict]] = {}
    folder_counts: List[tuple] = []  # (folder_name, count)

    # Sort subdirectories numerically when possible, fallback to lexicographic
    def sort_key(p: Path):
        """Sort numeric directory names before non-numeric names."""
        name = p.name
        return (0, int(name)) if name.isdigit() else (1, name)

    subdirs = sorted([p for p in root_dir.iterdir() if p.is_dir() and p.name.isdigit()], key=sort_key)

    # Create mapping from existing directory numbers to sequential 1-39
    directory_mapping = {}
    for i, subdir in enumerate(subdirs, 1):
        directory_mapping[subdir.name] = str(i)
        print(f"Mapping directory {subdir.name} → {i}")

    for subdir in subdirs:
        original_name = subdir.name
        mapped_name = directory_mapping[original_name]

        # Collect .txt files only, sorted lexicographically to keep a stable order
        txt_files = sorted([p for p in subdir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])

        # Track counts for reporting later
        folder_counts.append((original_name, len(txt_files)))

        entries: List[dict] = []
        for file_path in txt_files:
            content = file_path.read_text(encoding="utf-8")
            entries.append(
                {
                    "file_name": file_path.name,
                    "content": content,
                }
            )

        data[mapped_name] = entries

    return data, folder_counts


def main() -> None:
    """CLI entrypoint for building grouped dataset JSON from text files."""
    parser = argparse.ArgumentParser(
        description="Build a JSON file from text files grouped by id subdirectories inside ioi25."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Base directory; reads from base-dir/generated_datasets",
    )
    parser.add_argument(
        "--output-file-name",
        type=str,
        default="ioi25.json",
        help="Output JSON file name (written under --base-dir)",
    )

    args = parser.parse_args()

    data, folder_counts = collect_datasets(args.base_dir / "generated_datasets")

    output_file_path = args.base_dir / args.output_file_name
    with output_file_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Wrote JSON with {len(data)} ids to {output_file_path}")
    # Report per-folder counts
    for folder_name, count in folder_counts:
        print(f"{folder_name}: {count}")


if __name__ == "__main__":
    main()
