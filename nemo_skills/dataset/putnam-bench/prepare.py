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
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_URL = "https://github.com/trishullab/PutnamBench.git"

# Freeze to a specific commit - update this when needed
COMMIT_HASH = "64cedd86ef523f3d5f5dc7a21c10e3f69564c7d4"  # Latest commit as of 2025-10-23

LEAN_COMMENT_BLOCK_REGEX = re.compile(r"/--[\s\S]*?-/", re.MULTILINE)
LEAN_LINE_COMMENT_REGEX = re.compile(r"^\s*--.*$", re.MULTILINE)
LEAN_THEOREM_REGEX = re.compile(
    r"theorem\s+(putnam_[0-9]{4}_[ab][0-6])([\s\S]*?):=\s*(?:sorry|by)\b",
    re.MULTILINE,
)


def parse_lean_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")

    theorem_match = LEAN_THEOREM_REGEX.search(text)
    if not theorem_match:
        raise ValueError(f"No theorem found in {path}")

    theorem_name = theorem_match.group(1)
    theorem_body = theorem_match.group(2)
    theorem_text = f"theorem {theorem_name}{theorem_body}:= by\n"

    pre_theorem = text[: theorem_match.start()]

    doc_match = LEAN_COMMENT_BLOCK_REGEX.search(pre_theorem)
    line_match = LEAN_LINE_COMMENT_REGEX.search(pre_theorem)

    first_comment_start = None
    if doc_match:
        first_comment_start = doc_match.start()
    if line_match:
        first_comment_start = (
            line_match.start() if first_comment_start is None else min(first_comment_start, line_match.start())
        )

    if first_comment_start is None:
        header = pre_theorem.rstrip()
        informal_prefix = ""
    else:
        header = pre_theorem[:first_comment_start].rstrip()
        informal_prefix = pre_theorem[first_comment_start:].strip("\n")

    if informal_prefix:
        informal_prefix += "\n"

    header = (header.strip() + "\n") if header else ""

    result = {
        "name": theorem_name,
        "split": "test",
        "formal_statement": theorem_text,
        "informal_prefix": informal_prefix,
        "header": header,
    }
    return result


def download_dataset_and_process(output_path):
    output_dir = Path(output_path)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir) / "putnambench"
        # Clone the repo
        subprocess.run(["git", "clone", REPO_URL, str(repo_path)], check=True)

        # Get the current latest commit hash and freeze it, or use the hardcoded one
        global COMMIT_HASH
        if COMMIT_HASH is None:
            # Get the current latest commit
            result = subprocess.run(
                ["git", "-C", str(repo_path), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            COMMIT_HASH = result.stdout.strip()
        else:
            # Use the hardcoded commit hash
            subprocess.run(["git", "-C", str(repo_path), "checkout", COMMIT_HASH], check=True)

        # Run rewrite_solutions.py to generate solutions_replaced_new
        rewrite_script = repo_path / "lean4" / "scripts" / "rewrite_solutions.py"
        if not rewrite_script.exists():
            raise FileNotFoundError(f"rewrite_solutions.py not found at {rewrite_script}")

        subprocess.run([sys.executable, str(rewrite_script)], check=True, cwd=str(rewrite_script.parent))

        # Copy the processed lean files from solutions_replaced_new
        solutions_dir = repo_path / "lean4" / "solutions_replaced_new"
        if not solutions_dir.exists():
            raise FileNotFoundError(f"solutions_replaced_new directory not found at {solutions_dir}")

        for lean_file in solutions_dir.glob("*.lean"):
            shutil.copy(lean_file, output_dir / lean_file.name)


def delete_file(file_path):
    # delete the folder and all its contents
    path = Path(file_path)
    if path.exists():
        shutil.rmtree(path)


def main():
    data_dir = Path(__file__).absolute().parent
    original_folder = str(data_dir / "lean4")
    download_dataset_and_process(original_folder)

    # Extract data from processed lean files
    lean_files = sorted(Path(original_folder).glob("*.lean"))
    if len(lean_files) != 660:
        raise AssertionError(f"Expected 660 Lean files, found {len(lean_files)} in {original_folder}")

    entries = []
    for lean_file in lean_files:
        entry = parse_lean_file(lean_file)
        entries.append(entry)

    output_file = data_dir / "test.jsonl"
    with output_file.open("w", encoding="utf-8") as fout:
        for entry in entries:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    delete_file(original_folder)

    print(f"Processed {len(entries)} problems")
    print(f"Frozen to commit: {COMMIT_HASH}")


if __name__ == "__main__":
    main()
