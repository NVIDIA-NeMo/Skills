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
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from nemo_skills.dataset.utils import get_dataset_path


def parse_prepare_cli_arguments(args=None, datasets_nargs="+"):
    parser = argparse.ArgumentParser(description="Prepare all datasets")
    parser.add_argument("datasets", nargs=datasets_nargs, help="Specify one or more datasets to prepare")
    parser.add_argument(
        "--parallelism",
        type=int,
        default=20,
        help="Number of datasets to prepare in parallel",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries per dataset if preparation fails",
    )
    return parser.parse_known_args(args)


def prepare_datasets(
    datasets=None,
    extra_args="",
    parallelism=20,
    retries=3,
):
    datasets_dir = Path(__file__).absolute().parents[0]

    if not datasets:
        default_datasets = [d.name for d in datasets_dir.glob("*") if d.is_dir() and d.name != "__pycache__"]
        datasets = default_datasets

    max_workers = max(1, parallelism) if parallelism is not None else 1

    def run_prepare(dataset_name):
        dataset_path = get_dataset_path(dataset_name)
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
    args, unknown = parse_prepare_cli_arguments()
    extra_args = " ".join(unknown)

    prepare_datasets(
        args.datasets,
        extra_args=extra_args,
        parallelism=args.parallelism,
        retries=args.retries,
    )
