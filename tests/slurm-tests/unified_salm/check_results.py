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
import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # for utils.py
from utils import assert_all, soft_assert  # noqa: E402

_DIGIT_TO_WORD = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}


def normalize_text(text: str) -> str:
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^\w\s]", "", text)
    tokens = []
    for token in text.split():
        if token in _DIGIT_TO_WORD:
            tokens.append(_DIGIT_TO_WORD[token])
        else:
            tokens.append(token)
    return " ".join(tokens)


def load_references() -> dict[str, str]:
    container_path = Path("/nemo_run/code/tests/slurm-tests/unified_salm/salm_openai.test")
    local_path = Path(__file__).resolve().parent / "salm_openai.test"
    reference_path = container_path if container_path.exists() else local_path
    refs: dict[str, str] = {}
    with reference_path.open("rt", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            refs[row["id"]] = row["_reference"]
    return refs


def load_outputs(output_dir: Path) -> list[dict]:
    rows: list[dict] = []
    files = sorted(output_dir.glob("output*.jsonl"))
    soft_assert(len(files) > 0, f"No output JSONL files found in {output_dir}")
    for fpath in files:
        with fpath.open("rt", encoding="utf-8") as fin:
            for line in fin:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def check_salm_results(workspace: str):
    output_dir = Path(workspace) / "salm_outputs"
    soft_assert(output_dir.exists(), f"Missing output directory: {output_dir}")
    if not output_dir.exists():
        return

    references = load_references()
    rows = load_outputs(output_dir)

    soft_assert(len(rows) == len(references), f"Expected {len(references)} outputs, found {len(rows)}")

    for row in rows:
        sample_id = row.get("id")
        soft_assert(sample_id in references, f"Unexpected sample id in output: {sample_id}")
        if sample_id not in references:
            continue

        transcript = (row.get("generation") or "").strip()
        soft_assert(bool(transcript), f"Empty transcript for sample {sample_id}")
        if not transcript:
            continue

        ref_words = set(normalize_text(references[sample_id]).split())
        hyp_words = set(normalize_text(transcript).split())
        missing = sorted(ref_words - hyp_words)
        soft_assert(not missing, f"Sample {sample_id}: missing reference words: {', '.join(missing)}")

        debug_info = row.get("debug_info")
        if isinstance(debug_info, dict):
            soft_assert(
                debug_info.get("backend") == "salm",
                f"Sample {sample_id}: expected backend='salm' in debug_info, got {debug_info.get('backend')!r}",
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, help="Workspace directory containing results")
    args = parser.parse_args()

    check_salm_results(args.workspace)
    assert_all()


if __name__ == "__main__":
    main()
