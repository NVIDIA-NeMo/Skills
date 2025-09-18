# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import importlib
import subprocess
import sys
from pathlib import Path

from nemo_skills.dataset.utils import add_header_to_jsonl_inplace, get_lean4_header


def prepare_datasets(datasets=None, dataset_groups=None, add_lean4_header=False, extra_args=""):
    if datasets and dataset_groups:
        raise ValueError("Cannot specify both datasets and dataset_groups")

    datasets_dir = Path(__file__).absolute().parents[0]

    if not datasets:
        default_datasets = [d.name for d in datasets_dir.glob("*") if d.is_dir() and d.name != "__pycache__"]
        datasets = default_datasets

    if dataset_groups:
        target_datasets = []
        for dataset in datasets:
            dataset_module = importlib.import_module(f"nemo_skills.dataset.{dataset}")
            if getattr(dataset_module, "DATASET_GROUP", None) in dataset_groups:
                target_datasets.append(dataset)
        datasets = target_datasets

    for dataset in datasets:
        print(f"Preparing {dataset}")
        dataset_path = datasets_dir / dataset
        subprocess.run(f"{sys.executable} {dataset_path / 'prepare.py'} {extra_args}", shell=True, check=True)
        dataset_module = importlib.import_module(f"nemo_skills.dataset.{dataset}")

        if getattr(dataset_module, "DATASET_GROUP", None) == "math":
            if add_lean4_header:
                jsonl_files = list(dataset_path.glob("*.jsonl"))
                header = get_lean4_header()
                for jsonl_file in jsonl_files:
                    print(f"Adding Lean4 header to {jsonl_file}")
                    add_header_to_jsonl_inplace(jsonl_file, header)

    return datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare all datasets")
    parser.add_argument("datasets", nargs="+", help="Specify one or more datasets to prepare")
    parser.add_argument(
        "--dataset_groups",
        default=[],
        nargs="*",
        choices=["math", "code", "chat", "multichoice", "long-context", "tool"],
        help="Can specify a dataset group here",
    )
    parser.add_argument(
        "--add_lean4_header", action="store_true", help="Add Lean4 header to JSONL files during preparation"
    )
    args, unknown = parser.parse_known_args()
    extra_args = " ".join(unknown)

    prepare_datasets(args.datasets, args.dataset_groups, args.add_lean4_header, extra_args=extra_args)
