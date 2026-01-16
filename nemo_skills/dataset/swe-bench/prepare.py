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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--container_formatter",
        type=str,
        default="docker://swebench/sweb.eval.x86_64.{instance_id}",
        help="Container formatter string. You can download .sif containers and store them in a mounted "
        "directory which you can reference here to avoid redownloading all the time. "
        "See nemo_skills/dataset/swe-bench/dump_images.py",
    )
    parser.add_argument("--split", type=str, default="test", help="Swe-Bench dataset split to use")
    parser.add_argument(
        "--setup", type=str, default="default", help="Setup name (used as nemo-skills split parameter)."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="princeton-nlp/SWE-bench_Verified",
        help="Dataset name to load",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name
    split = args.split
    container_formatter = args.container_formatter

    dataset = datasets.load_dataset(path=dataset_name, split=split)
    output_file = Path(__file__).parent / f"{args.setup}.jsonl"
    dataset = dataset.map(lambda example: {**example, "container_formatter": container_formatter})
    dataset = dataset.add_column("container_id", list(range(len(dataset))))
    dataset = dataset.add_column("dataset_name", [dataset_name] * len(dataset))
    dataset = dataset.add_column("split", [split] * len(dataset))
    dataset.to_json(output_file, orient="records", lines=True)
