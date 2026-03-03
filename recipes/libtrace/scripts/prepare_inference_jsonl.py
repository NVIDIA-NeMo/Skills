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

"""
Prepare inference JSONL for applicability+relevance prompts.

Inputs: unified docs JSONL from LibTrace.
Outputs: JSONL with keys source, type, name, doc, domain.

Example:
  python /nemo_run/code/recipes/libtrace/scripts/prepare_inference_jsonl.py \
    --input_file /workspace/libtrace-results/harvest-docs-chem/results/chem_unified_docs.jsonl \
    --output_file /workspace/libtrace-results/prepare-inference-chem/results/chem_inference.jsonl \
    --domain chem
"""

from __future__ import annotations

import argparse
import json as _json_std
import random
from collections import defaultdict
from pathlib import Path

try:  # orjson is significantly faster; fallback to std json
    import orjson as _orjson  # type: ignore

    def _json_loads(s: str):
        return _orjson.loads(s)

    def _json_dumps(obj) -> str:
        return _orjson.dumps(obj).decode("utf-8")

except Exception:  # pragma: no cover - best effort
    _orjson = None

    def _json_loads(s: str):
        return _json_std.loads(s)

    def _json_dumps(obj) -> str:
        return _json_std.dumps(obj, ensure_ascii=False)


REQUIRED_KEYS = ("source", "type", "name", "doc")
DOMAIN_TO_LABEL = {
    "chem": "Chemistry",
    "chemistry": "Chemistry",
    "phys": "Physics",
    "physics": "Physics",
    "bio": "Biology",
    "biology": "Biology",
}


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if line == "":
                raise ValueError(f"Empty line in {path}:{line_num}")
            try:
                rows.append(_json_loads(line))
            except Exception as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_num}") from exc
    return rows


def resolve_domain_label(domain: str) -> str:
    return DOMAIN_TO_LABEL.get(domain.lower(), domain)


def validate_row(row: dict, path: Path, line_num: int, domain_label: str) -> dict:
    for key in REQUIRED_KEYS:
        if key not in row:
            raise KeyError(f"Missing '{key}' in {path}:{line_num}")

    source = row["source"]
    entry_type = row["type"]
    name = row["name"]
    doc = row["doc"]

    if not isinstance(source, str):
        raise TypeError(f"Expected 'source' to be str in {path}:{line_num}")
    if not isinstance(entry_type, str):
        raise TypeError(f"Expected 'type' to be str in {path}:{line_num}")
    if not isinstance(name, str):
        raise TypeError(f"Expected 'name' to be str in {path}:{line_num}")
    if not isinstance(doc, str):
        raise TypeError(f"Expected 'doc' to be str in {path}:{line_num}")

    return {
        "source": source,
        "type": entry_type,
        "name": name,
        "doc": doc,
        "domain": domain_label,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(_json_dumps(row) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LibTrace JSONL for inference.")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="JSONL file produced by LibTrace (from the harvest output dir).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to write the inference-ready JSONL file.",
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain identifier (e.g., chem, phys, bio) or a custom label.",
    )
    parser.add_argument(
        "--target_per_library",
        type=int,
        default=None,
        help="If set, sample up to this many entries per library.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for per-library sampling.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_path = Path(args.input_file)
    output_path = Path(args.output_file)
    domain_label = resolve_domain_label(args.domain)
    rows = read_jsonl(input_path)
    if not rows:
        raise RuntimeError(f"No entries found in {input_path}")

    output_rows: list[dict] = []
    for line_num, row in enumerate(rows, start=1):
        output_rows.append(validate_row(row, input_path, line_num, domain_label))

    if args.target_per_library is not None and args.target_per_library <= 0:
        raise ValueError("--target_per_library must be a positive integer.")

    if args.target_per_library:
        grouped: dict[str, list[dict]] = defaultdict(list)
        source_order: list[str] = []
        for row in output_rows:
            source = row["source"]
            if source not in grouped:
                source_order.append(source)
            grouped[source].append(row)

        rng = random.Random(args.seed)
        sampled_rows: list[dict] = []
        for source in source_order:
            items = grouped[source]
            if len(items) > args.target_per_library:
                indices = sorted(rng.sample(range(len(items)), k=args.target_per_library))
                items = [items[idx] for idx in indices]
            sampled_rows.extend(items)
        output_rows = sampled_rows

    write_jsonl(output_path, output_rows)
    print(f"Wrote {len(output_rows)} rows to {output_path}")
