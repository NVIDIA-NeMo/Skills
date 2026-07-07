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
Collect generated problems into a single JSONL.

Inputs: output-rs*.jsonl from problem generation.
Outputs: JSONL with problem, name, type, source, doc, [seed], source_file.

This script can optionally drop problems that exceed a token-length threshold
(useful for filtering broken generations that would exceed model context).

Example:
  python /nemo_run/code/recipes/libtrace/scripts/collect_generated_problems.py \
    --input_dir /workspace/libtrace-results/problem-generation-chem/results \
    --output_file /workspace/libtrace-results/collect-problems-chem/results/chem_problems.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json as _json_std
import re
from pathlib import Path

from transformers import AutoTokenizer  # type: ignore

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


def parse_seed(filename: str) -> int | None:
    match = re.search(r"output-rs(\d+)\.jsonl$", filename)
    if match:
        return int(match.group(1))
    return None


def read_jsonl(path: Path, skip_invalid: bool) -> tuple[list[dict], int]:
    rows: list[dict] = []
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if line == "":
                if skip_invalid:
                    skipped += 1
                    continue
                raise ValueError(f"Empty line in {path}:{line_num}")
            try:
                rows.append(_json_loads(line))
            except Exception as exc:
                if skip_invalid:
                    skipped += 1
                    continue
                raise ValueError(f"Invalid JSON in {path}:{line_num}") from exc
    return rows, skipped


def resolve_input_files(input_dir: str | None, input_files: list[str] | None, pattern: str) -> list[Path]:
    if input_dir is None and input_files is None:
        raise ValueError("Provide --input_dir or --input_files.")
    if input_dir is not None and input_files is not None:
        raise ValueError("Use only one of --input_dir or --input_files.")

    if input_dir is not None:
        glob_path = str(Path(input_dir) / pattern)
        files = sorted(Path(path) for path in glob.glob(glob_path))
        if not files:
            raise FileNotFoundError(f"No files matched {glob_path}")
        return files

    for path in input_files:
        if not Path(path).exists():
            raise FileNotFoundError(f"Input file not found: {path}")
    return [Path(path) for path in input_files]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect generated problems from output JSONL files.")
    parser.add_argument("--input_dir", type=str, default=None)
    parser.add_argument("--input_files", type=str, nargs="+", default=None)
    parser.add_argument("--pattern", type=str, default="output-rs*.jsonl")
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument(
        "--max_problem_tokens",
        type=int,
        default=10000,
        help="Drop problems whose tokenized length exceeds this threshold.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer name/path used for length filtering.",
    )
    parser.add_argument(
        "--max_dropped_examples",
        type=int,
        default=3,
        help="How many dropped (too-long) problems to print for debugging.",
    )
    parser.add_argument(
        "--skip_invalid",
        action="store_true",
        help="Skip invalid lines instead of failing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    input_files = resolve_input_files(args.input_dir, args.input_files, args.pattern)

    if args.max_problem_tokens is not None and args.max_problem_tokens <= 0:
        raise ValueError("--max_problem_tokens must be a positive integer or omitted.")
    if args.max_dropped_examples < 0:
        raise ValueError("--max_dropped_examples must be >= 0.")

    tokenizer = None
    if args.max_problem_tokens is not None:
        tokenizer_name = args.tokenizer
        if tokenizer_name is None:
            local_default = Path("/hf_models/gpt-oss-120b")
            tokenizer_name = str(local_default) if local_default.exists() else "openai/gpt-oss-120b"
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    collected = []
    total_skipped = 0
    dropped_too_long = 0
    printed_dropped = 0
    for path in input_files:
        rows, skipped = read_jsonl(path, args.skip_invalid)
        total_skipped += skipped
        seed = parse_seed(path.name)
        for line_num, row in enumerate(rows, start=1):
            missing = False
            for key in ("generation", "name", "type", "source", "doc"):
                if key not in row:
                    if args.skip_invalid:
                        total_skipped += 1
                        missing = True
                        break
                    raise KeyError(f"Missing '{key}' in {path}:{line_num}")
            if missing:
                continue

            problem = row["generation"]
            if not isinstance(problem, str):
                if args.skip_invalid:
                    total_skipped += 1
                    continue
                raise TypeError(f"Expected 'generation' to be str in {path}:{line_num}")
            problem = problem.strip()

            if tokenizer is not None:
                num_tokens = len(tokenizer.encode(problem, add_special_tokens=False))
                if num_tokens > args.max_problem_tokens:
                    dropped_too_long += 1
                    if printed_dropped < args.max_dropped_examples:
                        printed_dropped += 1
                        name = row["name"]
                        source = row["source"]
                        print(
                            f"\nDropped too-long problem ({num_tokens:,} tokens > {args.max_problem_tokens:,}) "
                            f"from {source} {name} in {path.name}:{line_num}\n"
                            f"{problem[:2000]}\n"
                            f"[... truncated, total_chars={len(problem):,}]"
                        )
                    continue

            item = {
                "problem": problem,
                "name": row["name"],
                "type": row["type"],
                "source": row["source"],
                "doc": row["doc"],
                "source_file": path.name,
            }
            if "domain" in row:
                item["domain"] = row["domain"]
            if seed is not None:
                item["seed"] = seed
            collected.append(item)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for item in collected:
            f.write(_json_dumps(item) + "\n")

    print(f"Wrote {len(collected)} items to {output_path}")
    if dropped_too_long:
        print(f"Dropped too-long problems: {dropped_too_long:,}")
    if total_skipped:
        print(f"Skipped invalid lines: {total_skipped}")
