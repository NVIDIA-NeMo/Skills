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
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/Goedel-LM/Goedel-Prover-V2/refs/heads/main/dataset/MOBench.jsonl"


def download_dataset(output_path: str):
    if not os.path.exists(output_path):
        urllib.request.urlretrieve(URL, output_path)


def load_jsonl(path: str):
    with open(path, "rt", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str, rows):
    with open(path, "wt", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row) + "\n")


def strip_trailing_sorry(text: str) -> str:
    # Normalize whitespace and remove trailing 'sorry' after ':= by'
    # Examples to handle:
    #   'theorem T : P := by sorry' -> 'theorem T : P := by'
    #   '... := by\n  sorry' -> '... := by'
    #   '... := by   sorry' -> '... := by'
    pattern = r"(?s)(:=\s*by)\s*sorry\s*$"
    return re.sub(pattern, r"\1", text.strip())


def split_prelude_and_theorem(code: str):
    """Split a Lean snippet into prelude (before theorem) and theorem block.

    Returns (prelude, theorem) or (None, None) if no theorem found.
    """
    m = re.search(r"(?m)^\s*theorem\s", code)
    if not m:
        return None, None
    prelude = code[: m.start()].strip()
    theorem = code[m.start() :].strip()
    return prelude, theorem


def extract_theorem_by(theorem_block: str) -> str:
    """Return theorem up to and including ':= by' and a trailing newline. Remove any trailing 'sorry'."""
    cleaned = strip_trailing_sorry(theorem_block)
    # Primary: capture everything up to ':= by'
    m = re.search(r"(?s)^(.*?:=\s*by)\b", cleaned)
    if m:
        base = m.group(1).rstrip()
    else:
        m2 = re.search(r":=\s*by", cleaned)
        if m2:
            base = cleaned[: m2.end()].rstrip()
        else:
            # If ':= by' is missing, append it to normalize format
            base = cleaned.rstrip() + " := by"
    # Ensure a single trailing newline
    return base + "\n"


def ensure_fields(entry: dict, lean_header: str) -> dict:
    """Normalize a MOBench entry to the minif2f-kimina JSON shape."""
    e = dict(entry)

    # Choose a candidate code block to split: prefer full_formal_statement, else formal_statement
    candidate_code = e.get("full_formal_statement") or e.get("formal_statement") or ""
    prelude, theorem_block = split_prelude_and_theorem(candidate_code)
    if not theorem_block:
        # If we couldn't find a theorem by regex, fall back to any present field
        theorem_block = candidate_code
    theorem_by = extract_theorem_by(theorem_block)

    # Keep any prelude (content before the 'theorem' declaration) as part of the
    # formal_statement, separated by a blank line from the theorem body.
    if prelude:
        prelude_clean = prelude.strip()
    else:
        prelude_clean = ""

    if prelude_clean:
        formal_statement = prelude_clean + "\n\n" + theorem_by
    else:
        formal_statement = theorem_by

    out = {
        "name": e.get("name", ""),
        "split": e.get("split", "test"),
        "informal_prefix": e.get("informal_prefix", ""),
        "formal_statement": formal_statement,
        "goal": "",
        "header": lean_header,
    }
    return out


def get_lean4_header() -> str:
    # Mirror minif2f/minif2f-kimina header
    return "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"


def main():
    data_dir = Path(__file__).absolute().parent
    original_file = str(data_dir / "MOBench.jsonl")
    test_file = str(data_dir / "test.jsonl")

    download_dataset(original_file)

    header = get_lean4_header()
    normalized_rows = (ensure_fields(entry, header) for entry in load_jsonl(original_file))
    write_jsonl(test_file, normalized_rows)

    # Clean up original
    try:
        os.remove(original_file)
    except OSError:
        pass


if __name__ == "__main__":
    main()
