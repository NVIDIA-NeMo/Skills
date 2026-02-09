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

# This is similar to nemo_skills/dataset/swe-bench/prepare.py, but adds an extra language column.

import argparse
from pathlib import Path

import datasets


# source: https://github.com/scaleapi/SWE-bench_Pro-os/blob/main/helper_code/image_uri.py
def get_docker_image_tag(uid, repo_name):
    repo_base, repo_name_only = repo_name.lower().split("/")
    hsh = uid.replace("instance_", "")

    if uid == "instance_element-hq__element-web-ec0f940ef0e8e3b61078f145f34dc40d1938e6c5-vnan":
        repo_name_only = "element-web"  # Keep full name for this one case
    elif "element-hq" in repo_name.lower() and "element-web" in repo_name.lower():
        repo_name_only = "element"
        if hsh.endswith("-vnan"):
            hsh = hsh[:-5]
    # All other repos: strip -vnan suffix
    elif hsh.endswith("-vnan"):
        hsh = hsh[:-5]

    tag = f"{repo_base}.{repo_name_only}-{hsh}"
    if len(tag) > 128:
        tag = tag[:128]

    return tag


def get_dockerhub_image_uri(uid, dockerhub_username, repo_name):
    return f"{dockerhub_username}/sweap-images:{get_docker_image_tag(uid, repo_name)}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--container_formatter",
        type=str,
        default="docker://{docker_tag}",
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
        default="ScaleAI/SWE-bench_Pro",
        help="Dataset name to load",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name
    split = args.split
    container_formatter = args.container_formatter
    assert "{docker_tag}" in container_formatter, "container_formatter must have {docker_tag}"

    dataset = datasets.load_dataset(path=dataset_name, split=split)
    output_file = Path(__file__).parent / f"{args.setup}.jsonl"

    dataset = dataset.rename_column("repo_language", "language")
    dataset = dataset.add_column(
        "container_formatter",
        [
            container_formatter.format(docker_tag=get_dockerhub_image_uri(row["instance_id"], "jefzda", row["repo"]))
            if container_formatter.startswith("docker://")
            else container_formatter.format(
                docker_tag=f"jefzda_sweap-images_{get_docker_image_tag(row['instance_id'], row['repo'])}"
            )
            for row in dataset
        ],
    )
    dataset = dataset.add_column("container_id", list(range(len(dataset))))
    dataset = dataset.add_column("dataset_name", [dataset_name] * len(dataset))
    dataset = dataset.add_column("split", [split] * len(dataset))
    dataset.to_json(output_file, orient="records", lines=True)
