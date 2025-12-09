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
from pathlib import Path

import datasets


def load_and_fix_dataset(dataset_name, split):
    print(f"Loading {dataset_name} (split: {split}) in streaming mode and dropping problematic columns...")
    ds_stream = datasets.load_dataset(path=dataset_name, split=split, streaming=True)

    def generator():
        columns_to_drop = [
            "fix_patch",
            "fixed_tests",
            "s2p_tests",
            "n2p_tests",
            "run_result",
            "test_patch_result",
            "fix_patch_result",
        ]

        for sample in ds_stream:
            for col in columns_to_drop:
                if col in sample:
                    del sample[col]

            yield sample

    # 3. Materialize the clean dataset
    return datasets.Dataset.from_generator(generator)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--container_formatter",
        type=str,
        default="docker://mswebench/sweb.eval.x86_64.{instance_id}",
        help="Container formatter string. You can download .sif containers and store them in a mounted "
        "directory which you can reference here to avoid redownloading all the time.",
    )  # TODO: add download script
    parser.add_argument("--split", type=str, default="train", help="dataset split to use")
    parser.add_argument("--setup", type=str, default="full", help="Setup name (used as nemo-skills split parameter).")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ByteDance-Seed/Multi-SWE-bench_mini",
        help="Dataset name to load",
        choices=[
            "ByteDance-Seed/Multi-SWE-bench_mini",
            "ByteDance-Seed/Multi-SWE-bench-flash",
            "ByteDance-Seed/Multi-SWE-bench",
        ],
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name
    split = args.split
    container_formatter = args.container_formatter

    # dataset = datasets.load_dataset(dataset_name, split=split)
    try:
        dataset = load_and_fix_dataset(dataset_name, split)
    except Exception as e:
        print(f"Standard loading failed. Error: {e}")
        raise e

    print(dataset)
    print("Column Names:", dataset.column_names)

    output_file = Path(__file__).parent / f"{args.setup}.jsonl"
    dataset = dataset.map(lambda example: {**example, "container_formatter": container_formatter})
    dataset = dataset.add_column("container_id", list(range(len(dataset))))
    dataset = dataset.add_column("dataset_name", [dataset_name] * len(dataset))
    dataset = dataset.add_column("split", [split] * len(dataset))
    dataset.to_json(output_file, orient="records", lines=True)
