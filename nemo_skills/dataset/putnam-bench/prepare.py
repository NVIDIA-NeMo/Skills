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

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO_URL = "https://github.com/trishullab/PutnamBench.git"


lean_regex = r"(^\s*theorem\s+([\S]+).+?sorry)"
lean_regex_match = re.compile(lean_regex, re.MULTILINE | re.DOTALL)
informal_prefix_regex = r"/--[\s\S]*?-/"
informal_prefix_match = re.compile(informal_prefix_regex)
header_regex = r"^(?:import|open|def|abbrev|noncomputable)\s+.*(?:\n(?:\s*\|.+|[ \t]+.+))*"
header_regex_match = re.compile(header_regex, re.MULTILINE)


def extract_theorem(filename):
    with open(filename, "r") as f:
        text = f.read()

    # retrieve the theorem name, formal statement, informal prefix, informal statement, and header
    lean_matches = lean_regex_match.findall(text)
    assert len(lean_matches) == 1, "Multiple theorems found in the file"
    informal_prefixes = informal_prefix_match.findall(text)
    assert len(informal_prefixes) == 1, "Multiple informal prefixes found in the file"
    headers = header_regex_match.findall(text)
    header = "\n".join(headers) + "\n\n"
    thm_name = lean_matches[0][1]
    full_thm = lean_matches[0][0]
    informal_prefix = informal_prefixes[0]
    informal_statement = informal_prefix.replace("/--", "").replace("-/", "").strip()

    theorem = {
        "name": thm_name,
        "formal_statement": full_thm,
        "informal_prefix": informal_prefix,
        "problem": informal_statement,
        "header": header,
    }

    return theorem


def download_dataset(output_path):
    output_dir = Path(output_path)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "putnambench"
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo_path)], check=True)
        src_dir = repo_path / "lean4" / "src"
        for lean_file in src_dir.rglob("putnam*.lean"):
            shutil.copy(lean_file, output_dir / lean_file.name)


def save_data(data, output_file):
    with open(output_file, "w", encoding="utf-8") as fout:
        for entry in data:
            fout.write(json.dumps(entry) + "\n")


def delete_file(file_path):
    # delete the folder and all its contents
    if os.path.exists(file_path):
        shutil.rmtree(file_path)


def main():
    data_dir = Path(__file__).absolute().parent
    original_folder = str(data_dir / "lean4")
    download_dataset(original_folder)

    # extract data
    theorems = []
    for filename in os.listdir(original_folder):
        if filename.endswith(".lean"):
            theorems.append(extract_theorem(os.path.join(original_folder, filename)))

    save_data(theorems, str(data_dir / "test.jsonl"))
    delete_file(original_folder)


if __name__ == "__main__":
    main()
