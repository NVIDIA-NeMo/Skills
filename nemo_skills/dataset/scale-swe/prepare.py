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
from pathlib import Path

import datasets


def prepare_example(example, container_formatter):
    """Add the field aliases consumed by the existing SWE-bench flow."""
    pre_commands = str(example.get("pre_commands") or "").strip()
    if pre_commands.endswith("\\n"):
        pre_commands = pre_commands[:-2].rstrip()
    repo = str(example["repo"])
    full_repo = f"{example['user']}/{repo}" if example.get("user") and "/" not in repo else repo
    return {
        "base_commit": example["parent_commit"],
        "repo": full_repo,
        "scale_swe_repo": repo,
        "container_formatter": container_formatter.format(
            image_url=example["image_url"],
            instance_id=example["instance_id"],
        ),
        "container_repo_dir": example["workdir"],
        "pre_commands": pre_commands,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--container_formatter",
        type=str,
        default="docker://{image_url}",
        help="Container formatter string. The default lets Apptainer pull each native Scale-SWE image. "
        "Use {image_url} and/or {instance_id} to reference fields from each sample.",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="PrimeIntellect/Scale-SWE-Verified",
        help="Dataset name to load",
    )
    parser.add_argument("--split", type=str, default="train", help="Scale-SWE dataset split to use")
    parser.add_argument(
        "--setup", type=str, default="default", help="Setup name (used as nemo-skills split parameter)."
    )
    args = parser.parse_args()

    dataset = datasets.load_dataset(path=args.dataset_name, split=args.split)
    dataset = dataset.map(
        lambda example: prepare_example(example, args.container_formatter),
        remove_columns=["parent_commit"],
    )
    dataset = dataset.add_column("container_id", list(range(len(dataset))))
    dataset = dataset.add_column("dataset_name", [args.dataset_name] * len(dataset))
    dataset = dataset.add_column("split", [args.split] * len(dataset))

    output_file = Path(__file__).parent / f"{args.setup}.jsonl"
    dataset.to_json(output_file, orient="records", lines=True)


if __name__ == "__main__":
    main()
