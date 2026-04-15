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

# Convert language codes to the same format as swe-bench-multilingual.
# This enables correct language-specific prompting and ignoring compilation files in git patches.
LANGUAGE_MAP = {
    "c": "c",
    "ts": "typescript",
    "go": "go",
    "python": "python",
}

# ScaleAI/SWE-Atlas-QnA has 124 tasks, associated with 11 unique docker images. We list them below with their linux distribution and version.
# ghcr.io/scaleapi/swe-atlas:swe_atlas_QnA_drakkan_sftpgo_1.0 -> alpine 3.18.3
# ghcr.io/scaleapi/swe-atlas:swe_atlas_QnA_grafana_k6_1.0 -> alpine 3.22.0
# ghcr.io/scaleapi/swe-atlas:swe_atlas_QnA_foxcpp_maddy_1.0 -> 11 debian
# ghcr.io/scaleapi/swe-atlas:swe_atlas_QnA_kovidgoyal_kitty_1.0 -> 24.04 ubuntu
# ghcr.io/scaleapi/swe-atlas:swe_atlas_QnA_paperless-ngx_paperless-ngx_e233ae8334038a4b615ea2e4ce663e30_qna_1.01 -> 11 debian
# ghcr.io/scaleapi/swe-atlas:swe_atlas_QnA_secdev_scapy_1.0 -> 12 debian
# ghcr.io/scaleapi/swe-atlas:swe_atlas_QnA_minio_minio_1.0 -> alpine 3.21.3
# ghcr.io/scaleapi/swe-atlas:swe_atlas_QnA_grafana_grafana_1.0 -> 12 debian
# ghcr.io/scaleapi/swe-atlas:swe_atlas_QnA_simple-login_app_1.0 -> 12 debian
# ghcr.io/scaleapi/swe-atlas:swe_atlas_QnA_Automattic_wp-calypso_1.0 -> 12 debian
# ghcr.io/scaleapi/swe-atlas:swe_atlas_QnA_trufflesecurity_trufflehog_1.0 -> 12 debian

# The following instances' dockerfiles are based on Alpine Linux (uses musl, not glibc).
# They have to be run in a separate eval job where the host Nemo-Skills container is also based on Alpine.
# Dockerfile: https://github.com/NVIDIA-NeMo/Skills/tree/main/dockerfiles/swe-bench/Dockerfile.nemo-skills.alpine
# This script creates separate dataset files for Alpine and Ubuntu instances.
ALPINE_INSTANCE_IDS = [
    "6905333b74f22949d97baa1b",
    "6905333b74f22949d97baa27",
    "6905333b74f22949d97baa23",
    "6905333b74f22949d97baa22",
    "6905333b74f22949d97baa1f",
    "6905333b74f22949d97baa25",
    "6905333b74f22949d97baa2c",
    "6905333b74f22949d97baa24",
    "6905333b74f22949d97baa1a",
    "6905333b74f22949d97baa1c",
    "6905333b74f22949d97baa26",
    "6905333b74f22949d97baa20",
    "6905333b74f22949d97baa28",
    "6905333b74f22949d97baa1e",
    "6905333b74f22949d97baa2d",
    "6905333b74f22949d97baa2b",
    "6905333b74f22949d97ba9bb",
    "6905333b74f22949d97baa2a",
    "6905333b74f22949d97ba9f1",
    "6905333b74f22949d97ba9c0",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--container_formatter",
        type=str,
        default="docker://{docker_image}",
        help="Container formatter string. You can download .sif containers and store them in a mounted "
        "directory which you can reference here to avoid redownloading all the time. "
        "See nemo_skills/dataset/swe-bench/dump_images.py",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ScaleAI/SWE-Atlas-QnA",
        help="Dataset name to load",
    )
    parser.add_argument("--split", type=str, default="test", help="Swe-Bench dataset split to use")
    parser.add_argument(
        "--setup",
        type=str,
        default="default",
        help="Setup name. Creates two dataset files: <setup>.alpine.jsonl and <setup>.ubuntu.jsonl",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name
    split = args.split
    container_formatter = args.container_formatter
    assert "{docker_image}" in container_formatter, "container_formatter must have {docker_image}"

    dataset = datasets.load_dataset(path=dataset_name, split=split)
    output_file_alpine = Path(__file__).parent / f"{args.setup}.alpine.jsonl"
    output_file_ubuntu = Path(__file__).parent / f"{args.setup}.ubuntu.jsonl"

    dataset = dataset.rename_column("prompt", "problem_statement")
    dataset = dataset.rename_column("task_id", "instance_id")
    dataset = dataset.map(lambda x: {"language": LANGUAGE_MAP[x["language"]]})

    dataset = dataset.add_column(
        "container_formatter",
        [
            container_formatter.format(docker_image=row["docker_image"])
            if container_formatter.startswith("docker://")
            else container_formatter.format(docker_image=row["docker_image"].replace("/", "_").replace(":", "_"))
            for row in dataset
        ],
    )

    dataset = dataset.add_column("container_id", list(range(len(dataset))))
    dataset = dataset.add_column("dataset_name", [dataset_name] * len(dataset))
    dataset = dataset.add_column("split", [split] * len(dataset))
    dataset = dataset.add_column("container_repo_dir", ["/app"] * len(dataset))

    alpine_dataset = dataset.filter(lambda x: x["instance_id"] in ALPINE_INSTANCE_IDS)
    alpine_dataset.to_json(output_file_alpine, orient="records", lines=True)
    ubuntu_dataset = dataset.filter(lambda x: x["instance_id"] not in ALPINE_INSTANCE_IDS)
    ubuntu_dataset.to_json(output_file_ubuntu, orient="records", lines=True)
