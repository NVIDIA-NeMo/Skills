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
import os
import urllib.request
from pathlib import Path

# Source: Goedel-LM/Goedel-Prover-V2 minif2f dataset
URL = "https://raw.githubusercontent.com/Goedel-LM/Goedel-Prover-V2/refs/heads/main/dataset/minif2f.jsonl"


def download_dataset(output_path):
    if not os.path.exists(output_path):
        urllib.request.urlretrieve(URL, output_path)


def _ensure_header_ends_with_by(text: str) -> str:
    # Ensure the snippet ends with " := by\n" and nothing after it
    marker = ":= by"
    idx = text.rfind(marker)
    if idx != -1:
        return text[: idx + len(marker)] + "\n"
    return text


def clean_lean_snippet(text: str | None) -> str | None:
    if text is None:
        return None
    # Remove all occurrences of "sorry" variants first
    cleaned = text.replace(" by sorry", " by").replace("by sorry", "by").replace("sorry", "")
    # Then enforce DeepSeek-style header ending
    cleaned = _ensure_header_ends_with_by(cleaned)
    return cleaned


def _split_header_and_theorem(text: str) -> tuple[str, str]:
    # Try to split header (imports, options, opens) from the theorem block.
    # Heuristic: header precedes the first occurrence of "/--" (doc) or "theorem ".
    header_end = -1
    doc_idx = text.find("/--")
    if doc_idx != -1:
        header_end = doc_idx
    thm_idx = text.find("theorem ")
    if header_end == -1 or (thm_idx != -1 and thm_idx < header_end):
        header_end = thm_idx
    if header_end is None or header_end <= 0:
        header = ""
    else:
        header = text[:header_end]

    # The theorem starts from the first "theorem " occurrence if present
    if thm_idx != -1:
        theorem = text[thm_idx:]
    else:
        theorem = text
    return header, theorem


def process_entry(entry: dict) -> dict:
    # Build an output entry that mirrors the DeepSeek minif2f fields and order.
    name = entry.get("name", "")
    split = entry.get("split", entry.get("category", ""))
    informal_prefix = entry.get("informal_prefix", "")

    # Prefer formal_statement if present, else fall back to lean4_code
    raw_code = entry.get("formal_statement") or entry.get("lean4_code") or ""
    header, theorem = _split_header_and_theorem(raw_code)
    header = header
    theorem = clean_lean_snippet(theorem) or ""

    out: dict = {}
    out["name"] = name
    if split:
        out["split"] = split
    out["informal_prefix"] = informal_prefix
    out["formal_statement"] = theorem
    # The DeepSeek format includes a goal; we don't synthesize it, keep empty
    out["goal"] = entry.get("goal", "")
    out["header"] = header
    return out


def split_data(input_file):
    valid_data = []
    test_data = []

    with open(input_file, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            entry = json.loads(line)
            entry = process_entry(entry)
            split_value = entry["split"]  # Will raise KeyError if missing
            if split_value == "valid":
                valid_data.append(entry)
            elif split_value == "test":
                test_data.append(entry)
            else:
                raise ValueError(f"Unknown split value: {split_value!r} in entry: {entry}")

    return valid_data, test_data


def save_data(data, output_file):
    with open(output_file, "w", encoding="utf-8") as fout:
        for entry in data:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")


def delete_file(file_path):
    if os.path.exists(file_path):
        os.remove(file_path)


def main(split):
    data_dir = Path(__file__).absolute().parent
    original_file = str(data_dir / "minif2f-kimina.jsonl")
    valid_file = str(data_dir / "valid.jsonl")
    test_file = str(data_dir / "test.jsonl")

    download_dataset(original_file)
    valid_data, test_data = split_data(original_file)

    if split == "valid":
        save_data(valid_data, valid_file)
    elif split == "test":
        save_data(test_data, test_file)
    elif split == "all":
        save_data(valid_data, valid_file)
        save_data(test_data, test_file)

    delete_file(original_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="all", choices=("all", "test", "valid"), help="Data split to process")
    args = parser.parse_args()

    main(args.split)
