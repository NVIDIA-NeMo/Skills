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
import json
from pathlib import Path

import datasets


def get_date_range(start_str, end_str):
    """Generates a list of YYYY_MM strings between start and end inclusive."""
    start_year, start_month = map(int, start_str.split("_"))
    end_year, end_month = map(int, end_str.split("_"))

    dates = []
    current_year, current_month = start_year, start_month

    while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
        dates.append(f"{current_year}_{current_month:02d}")

        current_month += 1
        if current_month > 12:
            current_month = 1
            current_year += 1
    return dates


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--container_formatter",
        type=str,
        default="docker://{docker_image}",
        help="Container formatter string. You can download .sif containers and store them in a mounted "
        "directory which you can reference here to avoid redownloading all the time.",
    )
    parser.add_argument("--start_date", type=str, help="Start date in YYYY_MM format")
    parser.add_argument("--end_date", type=str, help="End date in YYYY_MM format")
    parser.add_argument(
        "--setup", type=str, default="default", help="Setup name (used as nemo-skills split parameter)."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="nebius/SWE-rebench-leaderboard",
        help="Dataset name to load",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name
    output_file = Path(__file__).parent / f"{args.setup}.jsonl"

    if args.start_date and args.end_date:
        splits_to_load = get_date_range(args.start_date, args.end_date)
    else:
        print("Start/End date not provided. Defaulting to 'test' split.")
        splits_to_load = ["test"]

    all_data = []
    global_id_counter = 0

    for split in splits_to_load:
        print(f"Loading split: {split}...")
        try:
            ds = datasets.load_dataset(path=dataset_name, split=split)
            for item in ds:
                docker_image = item["docker_image"]
                if args.container_formatter.endswith(".sif"):
                    docker_image = item["docker_image"].replace("/", "_").replace(":", "_")
                container_formatter = args.container_formatter.format(docker_image=docker_image)
                processed_item = {
                    **item,
                    "container_formatter": container_formatter,
                    "container_id": global_id_counter,
                    "dataset_name": dataset_name,
                    "split": split,
                }
                all_data.append(processed_item)
                global_id_counter += 1

        except Exception as e:
            print(f"Warning: Could not load split {split}. Error: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in all_data:
            f.write(json.dumps(entry) + "\n")

    print(f"Successfully saved {len(all_data)} samples to {output_file}")
