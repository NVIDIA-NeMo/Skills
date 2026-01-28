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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from nemo_skills.dataset.utils import add_header_to_jsonl_inplace, get_lean4_header


def prepare_datasets(
    datasets=None,
    dataset_groups=None,
    add_lean4_header=False,
    extra_args="",
    parallelism=20,
    retries=3,
):
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

    max_workers = max(1, parallelism) if parallelism is not None else 1

    def run_prepare(dataset_name):
        dataset_path = datasets_dir / dataset_name
        attempts = max(1, retries + 1)
        for attempt in range(1, attempts + 1):
            if attempts > 1:
                print(f"Preparing {dataset_name} (attempt {attempt}/{attempts})")
            else:
                print(f"Preparing {dataset_name}")
            try:
                subprocess.run(
                    f"{sys.executable} {dataset_path / 'prepare.py'} {extra_args}",
                    shell=True,
                    check=True,
                )
                break
            except subprocess.CalledProcessError:
                if attempt == attempts:
                    raise
                print(f"Retrying {dataset_name} after failure")

        dataset_module = importlib.import_module(f"nemo_skills.dataset.{dataset_name}")
        if getattr(dataset_module, "DATASET_GROUP", None) == "math" and add_lean4_header:
            jsonl_files = list(dataset_path.glob("*.jsonl"))
            header = get_lean4_header()
            for jsonl_file in jsonl_files:
                print(f"Adding Lean4 header to {jsonl_file}")
                add_header_to_jsonl_inplace(jsonl_file, header)
        return dataset_name

    errors = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_prepare, dataset): dataset for dataset in datasets}
        for future in as_completed(futures):
            dataset = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                errors.append((dataset, exc))

    if errors:
        first_dataset, first_error = errors[0]
        raise RuntimeError(f"Failed to prepare dataset {first_dataset}") from first_error

    return list(datasets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare all datasets")
    parser.add_argument("datasets", nargs="+", help="Specify one or more datasets to prepare")
    parser.add_argument(
        "--dataset_groups",
        default=[],
        nargs="*",
        choices=["math", "code", "chat", "multichoice", "long-context", "tool", "vlm"],
        help="Can specify a dataset group here",
    )
    parser.add_argument(
        "--add_lean4_header", action="store_true", help="Add Lean4 header to JSONL files during preparation"
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=20,
        help="Number of datasets to prepare in parallel",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Number of retries per dataset if preparation fails",
    )
    args, unknown = parser.parse_known_args()
    extra_args = " ".join(unknown)

    prepare_datasets(
        args.datasets,
        args.dataset_groups,
        args.add_lean4_header,
        extra_args=extra_args,
        parallelism=args.parallelism,
        retries=args.retries,
    )
