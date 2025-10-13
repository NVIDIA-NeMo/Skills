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

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://huggingface.co/datasets/He-Ren/OJBench_testdata"
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("❌ Error: Hugging Face token not found.", file=sys.stderr)
    print("   Please set the HF_TOKEN environment variable with your access token.", file=sys.stderr)
    print("   You can create a token at: https://huggingface.co/settings/tokens", file=sys.stderr)
    sys.exit(1)


def clone_dataset_repo(url, destination):
    if not shutil.which("git"):
        print("❌ Error: Git executable not found. Please install Git.", file=sys.stderr)
        sys.exit(1)

    try:
        if destination.exists() or destination.is_symlink():
            print(f"Destination '{destination}' already exists. Removing it...")
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()

        auth_url = url.replace("https://huggingface.co/", f"https://user:{HF_TOKEN}@huggingface.co/", 1)
        print(f"Cloning {url} into {destination}...")
        subprocess.run(["git", "clone", auth_url, destination], check=True, capture_output=True)

        print("✅ Git clone is successful.")

    except subprocess.CalledProcessError as e:
        print("❌ Git command failed:", file=sys.stderr)
        cmd = [url if i == 2 else arg for i, arg in enumerate(e.cmd)]
        print(f"   Command: {' '.join(map(str, cmd))}", file=sys.stderr)
        stderr = e.stderr.decode().strip()
        stderr = stderr.replace(HF_TOKEN, "***") if HF_TOKEN else stderr
        print(f"   Stderr: {stderr}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    destination = data_dir / "OJBench_testdata"
    clone_dataset_repo(REPO_URL, destination)

    source_file = destination / "prompts" / "full.jsonl"
    python_target_file = data_dir / "test_python.jsonl"
    cpp_target_file = data_dir / "test_cpp.jsonl"

    print(f"Processing '{source_file}' and splitting into Python and C++ subsets...")
    processed_lines = 0
    try:
        with (
            source_file.open("r", encoding="utf-8") as infile,
            python_target_file.open("w", encoding="utf-8") as outfile_py,
            cpp_target_file.open("w", encoding="utf-8") as outfile_cpp,
        ):
            for line in infile:
                data = json.loads(line)
                data["question"] = data.pop("prompt")
                data["subset_for_metrics"] = data["difficulty"]
                if data["language"] == "python":
                    outfile_py.write(json.dumps(data) + "\n")
                elif data["language"] == "cpp":
                    outfile_cpp.write(json.dumps(data) + "\n")
                processed_lines += 1
        print(f"✅ Successfully processed {processed_lines} lines.")

    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        print(f"❌ Error during file processing: {e}", file=sys.stderr)
        sys.exit(1)
