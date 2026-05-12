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
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed


def read_repos(jsonl_file):
    """Read all repos from a jsonl file."""
    repos = set()

    with open(jsonl_file, "r") as f:
        for line in f:
            data = json.loads(line)
            repos.add(data["repo"])

    return sorted(repos)


def clone_repo(repo, output_dir, force):
    """Clone a GitHub repo."""
    repo_dir = os.path.join(output_dir, repo)

    # Check if cloned repo already exists
    if os.path.exists(repo_dir):
        if not force:  # skip by default unless --force was set
            print(f"✓ {repo} -> {repo_dir} (already exists)")
            return True
        else:
            print(f"{repo_dir} already exists, removing because --force was set...")
            shutil.rmtree(repo_dir)

    try:
        # Clone repo
        cmd = ["git", "clone", "--mirror", f"https://github.com/{repo}", repo_dir]

        print(f"Cloning {repo}...")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print(f"✓ {repo} -> {repo_dir}")
            return True
        else:
            print(f"✗ Failed to clone {repo}")
            print(f"  Error: {result.stderr.strip()}")
            return False

    except Exception as e:
        print(f"✗ Error cloning {repo}: {e}")
        return False


def main():
    """Parse command-line arguments using argparse."""
    parser = argparse.ArgumentParser(description="Download repos from a JSONL dataset file.")

    parser.add_argument("input_file", help="JSONL file to read repos from (must have 'repo' field)")
    parser.add_argument("output_directory", help="Directory to save repos")
    parser.add_argument("--max_workers", "-j", type=int, default=20, help="Number of parallel downloads (default: 20)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing repos")

    args = parser.parse_args()

    jsonl_path = args.input_file
    output_dir = args.output_directory
    max_workers = args.max_workers
    force = args.force

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Read repos from JSONL
    if not os.path.exists(jsonl_path):
        print(f"Error: {jsonl_path} not found!")
        sys.exit(1)

    print(f"Reading repos from {jsonl_path}...")
    repos = read_repos(jsonl_path)

    print(f"Found {len(repos)} unique repos to download.")
    print(f"Output directory: {output_dir}")
    print(f"Using {max_workers} parallel workers.\n")

    # Download repos in parallel
    successful = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        futures = [executor.submit(clone_repo, repo, output_dir, force) for repo in repos]

        # Process completed tasks
        for future in as_completed(futures):
            success = future.result()
            if success:
                successful += 1
            else:
                failed += 1

    print("\nDownload complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(repos)}")


if __name__ == "__main__":
    main()
