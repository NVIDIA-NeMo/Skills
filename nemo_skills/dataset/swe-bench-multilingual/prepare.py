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

# SWE-bench Multilingual repositories
# Source: https://www.swebench.com/multilingual.html#appendix-a
REPO_TO_LANGUAGE = {
    "redis/redis": "c",
    "jqlang/jq": "c",
    "nlohmann/json": "cpp",
    "micropython/micropython": "c",
    "valkey-io/valkey": "c",
    "fmtlib/fmt": "cpp",
    "caddyserver/caddy": "go",
    "hashicorp/terraform": "go",
    "prometheus/prometheus": "go",
    "gohugoio/hugo": "go",
    "gin-gonic/gin": "go",
    "google/gson": "java",
    "apache/druid": "java",
    "projectlombok/lombok": "java",
    "apache/lucene": "java",
    "reactivex/rxjava": "java",
    "javaparser/javaparser": "java",
    "babel/babel": "javascript",
    "vuejs/core": "javascript",
    "facebook/docusaurus": "javascript",
    "immutable-js/immutable-js": "javascript",
    "mrdoob/three.js": "javascript",
    "preactjs/preact": "javascript",
    "axios/axios": "javascript",
    "phpoffice/phpspreadsheet": "php",
    "laravel/framework": "php",
    "php-cs-fixer/php-cs-fixer": "php",
    "briannesbitt/carbon": "php",
    "jekyll/jekyll": "ruby",
    "fluent/fluentd": "ruby",
    "fastlane/fastlane": "ruby",
    "jordansissel/fpm": "ruby",
    "faker-ruby/faker": "ruby",
    "rubocop/rubocop": "ruby",
    "tokio-rs/tokio": "rust",
    "uutils/coreutils": "rust",
    "nushell/nushell": "rust",
    "tokio-rs/axum": "rust",
    "burntsushi/ripgrep": "rust",
    "sharkdp/bat": "rust",
    "astral-sh/ruff": "rust",
}


def get_language(row):
    repo = row["repo"]
    if repo in REPO_TO_LANGUAGE:
        return REPO_TO_LANGUAGE[repo]
    else:
        print(
            f"Warning: programming language could not be inferred for unknown repo {repo}. "
            "Setting the 'language' column to 'unknown'. It is recommended to add language labels for optimal scores."
        )
        return "unknown"


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
        default="SWE-bench/SWE-bench_Multilingual",
        help="Dataset name to load",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name
    split = args.split
    container_formatter = args.container_formatter

    dataset = datasets.load_dataset(path=dataset_name, split=split)
    output_file = Path(__file__).parent / f"{args.setup}.jsonl"

    # For the standard SWE-bench Multilingual dataset, we add a language label based on the repository name.
    # For custom datasets, this won't work, so we warn the user that for optimal scores they should add their own labels.

    if "language" in dataset.column_names:
        print("Dataset already has a 'language' column. Language labels will not be added.")
    elif "repo" in dataset.column_names:
        print("Adding programming language labels...")
        dataset = dataset.map(lambda row: {**row, "language": get_language(row)})
    else:
        print(
            "Warning: programming language labels could not be inferred for this dataset "
            "because there is no 'language' or 'repo' column. "
            "Setting the 'language' column to 'unknown'. It is recommended to add language labels for optimal scores."
        )
        dataset = dataset.add_column("language", ["unknown"] * len(dataset))

    dataset = dataset.add_column("container_formatter", [container_formatter] * len(dataset))
    dataset = dataset.add_column("container_id", list(range(len(dataset))))
    dataset = dataset.add_column("dataset_name", [dataset_name] * len(dataset))
    dataset = dataset.add_column("split", [split] * len(dataset))
    dataset.to_json(output_file, orient="records", lines=True)
