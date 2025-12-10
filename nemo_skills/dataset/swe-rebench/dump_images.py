# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


def read_container_names(jsonl_file):
    """Read all container names from the SWE-bench test.jsonl file."""
    container_names = set()

    with open(jsonl_file, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            # Use the same logic as in swebench.py
            container_formatter = data["container_formatter"]
            instance_id = data["instance_id"]

            container_name = container_formatter.format(instance_id=instance_id.replace("__", "_1776_"))
            container_names.add(container_name)

    return sorted(list(container_names))


def convert_to_sif(container_name, output_dir):
    """Convert a Docker container to SIF format using apptainer."""
    # Create a safe filename for the SIF file
    container_name = container_name.removeprefix("docker://")
    safe_name = container_name.replace("/", "_").replace(":", "_")
    sif_path = os.path.join(output_dir, f"{safe_name}.sif")

    # Check if SIF file already exists
    if os.path.exists(sif_path):
        print(f"✓ {container_name} -> {sif_path} (already exists)")
        return True, container_name, sif_path

    try:
        # Use apptainer to build SIF from Docker container
        cmd = ["apptainer", "build", sif_path, f"docker://{container_name}"]

        print(f"Converting {container_name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✓ {container_name} -> {sif_path}")
            return True, container_name, sif_path
        else:
            print(f"✗ Failed to convert {container_name}")
            print(f"  Error: {result.stderr.strip()}")
            return False, container_name, None

    except Exception as e:
        print(f"✗ Error converting {container_name}: {e}")
        return False, container_name, None


def main():
    """Parse command-line arguments using argparse."""
    parser = argparse.ArgumentParser(
        description="Convert Docker containers from a JSONL test file into Apptainer SIF images."
    )

    parser.add_argument("input_file", help="JSONL file to read container URLs from (SWE-bench format)")
    parser.add_argument("output_directory", help="Directory to save SIF files")
    parser.add_argument(
        "--max-workers", "-j", type=int, default=20, help="Number of parallel conversions (default: 20)"
    )

    args = parser.parse_args()

    jsonl_path = args.input_file
    output_dir = args.output_directory
    max_workers = args.max_workers

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read container names from JSONL
    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} not found!")
        sys.exit(1)

    print(f"Reading container names from {jsonl_path}...")
    container_names = read_container_names(jsonl_path)

    print(f"Found {len(container_names)} unique containers to convert.")
    print(f"Output directory: {output_dir}")
    print(f"Using {max_workers} parallel workers.\n")

    # Convert containers in parallel
    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all conversion tasks
        future_to_container = {
            executor.submit(convert_to_sif, container_name, output_dir): container_name
            for container_name in container_names
        }

        # Process completed tasks
        for future in as_completed(future_to_container):
            success, container_name, sif_path = future.result()
            if success:
                successful += 1
            else:
                failed += 1

    print("\nConversion complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(container_names)}")


if __name__ == "__main__":
    main()
