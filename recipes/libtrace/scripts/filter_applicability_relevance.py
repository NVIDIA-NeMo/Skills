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
Filter applicability+relevance outputs and report label distributions.

Inputs: output-rs*.jsonl from the labeling step.
Outputs: filtered JSONL with *_applicable and *_relevance fields.
The filtered output keeps only the original LibTrace fields (source/type/name/doc/domain) plus labels.

Example:
  python /nemo_run/code/recipes/libtrace/scripts/filter_applicability_relevance.py \
    --input_file /workspace/libtrace-results/applicability-relevance-chem/results/output.jsonl \
    --output_file /workspace/libtrace-results/filter-applicability-relevance-chem/results/chem_filtered.jsonl \
    --domain chem --require_applicable --min_relevance 3
"""

from __future__ import annotations

import argparse
import json as _json_std
import re
from collections import Counter
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


DOMAIN_TO_LABEL = {
    "chem": "Chemistry",
    "chemistry": "Chemistry",
    "phys": "Physics",
    "physics": "Physics",
    "bio": "Biology",
    "biology": "Biology",
}

KEEP_FIELDS = ("source", "type", "name", "doc", "domain")


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


def extract_generation_text(item: dict) -> str:
    generation = item.get("generation")
    if isinstance(generation, str):
        return generation

    serialized = item.get("serialized_output")
    if isinstance(serialized, list) and serialized:
        first = serialized[0]
        if isinstance(first, dict):
            content = first.get("content")
            if isinstance(content, str):
                return content

    raise KeyError("Missing generation text (expected 'generation' or 'serialized_output[0].content').")


def parse_scores(generation: str, label: str, path: str, line_num: int) -> tuple[int, int]:
    applicable_pattern = re.compile(rf"{re.escape(label)}-Applicable:\s*([01])")
    relevance_pattern = re.compile(rf"{re.escape(label)}-Relevance:\s*([123])")

    applicable_match = applicable_pattern.search(generation)
    if applicable_match is None:
        raise ValueError(f"Missing {label}-Applicable in {path}:{line_num}")
    applicable = int(applicable_match.group(1))

    relevance_match = relevance_pattern.search(generation)
    if relevance_match is None:
        raise ValueError(f"Missing {label}-Relevance in {path}:{line_num}")
    relevance = int(relevance_match.group(1))

    return applicable, relevance


def resolve_label(domain: str | None, label: str | None) -> str:
    if domain is None and label is None:
        raise ValueError("Provide --domain or --label.")
    if domain is not None and label is not None:
        raise ValueError("Use only one of --domain or --label.")
    if label is not None:
        return label
    domain_key = domain.lower()
    if domain_key not in DOMAIN_TO_LABEL:
        raise ValueError(f"Unsupported domain: {domain}")
    return DOMAIN_TO_LABEL[domain_key]


def print_distribution(title: str, counter: Counter, keys: list[int] | None = None) -> None:
    total = sum(counter.values())
    print(title)
    if total == 0:
        print("  (none)")
        return
    if keys is None:
        keys = sorted(counter.keys())
    for key in keys:
        count = counter.get(key, 0)
        pct = (count / total) * 100
        print(f"  {key}: {count:,} ({pct:.2f}%)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter LibTrace applicability+relevance outputs.")
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument(
        "--key_prefix",
        type=str,
        default=None,
        help="Prefix for output keys (default: domain or label lowercased).",
    )
    parser.add_argument(
        "--require_applicable",
        action="store_true",
        default=True,
        help="Require Applicable=1 (default: True).",
    )
    parser.add_argument(
        "--no_require_applicable",
        action="store_false",
        dest="require_applicable",
    )
    parser.add_argument("--min_relevance", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument(
        "--max_relevance",
        type=int,
        default=3,
        choices=[1, 2, 3],
        help="Optional upper bound for relevance (default: 3).",
    )
    parser.add_argument(
        "--skip_invalid",
        action="store_true",
        help="Skip rows with missing or malformed labels instead of failing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    label = resolve_label(args.domain, args.label)
    key_prefix = args.key_prefix or (args.domain or label).lower()

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    data, skipped_read = read_jsonl(input_path, args.skip_invalid)
    if not data:
        raise RuntimeError(f"No entries found in {input_path}")

    source_dist_pre: Counter[str] = Counter()
    valid_items: list[dict] = []
    invalid_count = skipped_read
    for line_num, item in enumerate(data, start=1):
        try:
            source = item["source"]
            if not isinstance(source, str):
                raise TypeError
            source_dist_pre[source] += 1
            valid_items.append(item)
        except Exception:
            if not args.skip_invalid:
                raise ValueError(f"Missing or invalid source in {input_path}:{line_num}")
            invalid_count += 1

    if not valid_items:
        raise RuntimeError("No valid entries after filtering invalid rows.")

    print(f"\nSource distribution (pre-filtering) for {label}:")
    for source, count in sorted(source_dist_pre.items(), key=lambda x: -x[1]):
        print(f"  {source}: {count:,} ({count / len(valid_items) * 100:.2f}%)")
    print(f"Total unique sources: {len(source_dist_pre)}")
    print(f"Total samples: {len(valid_items):,}")

    applicable_counts: Counter[int] = Counter()
    relevance_counts: Counter[int] = Counter()
    relevance_counts_applicable: Counter[int] = Counter()
    pair_counts: Counter[tuple[int, int]] = Counter()
    filtered_samples = []

    for line_num, item in enumerate(valid_items, start=1):
        try:
            generation = extract_generation_text(item)
            applicable, relevance = parse_scores(generation, label, str(input_path), line_num)
        except Exception:
            if not args.skip_invalid:
                raise
            invalid_count += 1
            continue

        applicable_counts[applicable] += 1
        relevance_counts[relevance] += 1
        pair_counts[(applicable, relevance)] += 1
        if applicable == 1:
            relevance_counts_applicable[relevance] += 1

        passes_applicable = applicable == 1 if args.require_applicable else True
        if passes_applicable and args.min_relevance <= relevance <= args.max_relevance:
            filtered_item = {}
            for key in KEEP_FIELDS:
                if key in item:
                    filtered_item[key] = item[key]
            filtered_item[f"{key_prefix}_applicable"] = applicable
            filtered_item[f"{key_prefix}_relevance"] = relevance
            filtered_samples.append(filtered_item)

    print("\nLabel distributions:")
    print_distribution("Applicable:", applicable_counts, keys=[0, 1])
    print_distribution("Relevance (all):", relevance_counts, keys=[1, 2, 3])
    print_distribution("Relevance (Applicable=1):", relevance_counts_applicable, keys=[1, 2, 3])
    print("\nApplicable/Relevance pairs:")
    for applicable in [0, 1]:
        for relevance in [1, 2, 3]:
            count = pair_counts.get((applicable, relevance), 0)
            print(f"  ({applicable}, {relevance}): {count:,}")

    if invalid_count:
        print(f"\nSkipped invalid rows: {invalid_count}")

    if args.require_applicable:
        summary = f"{label}-Applicable=1 and {label}-Relevance in [{args.min_relevance}, {args.max_relevance}]"
    else:
        summary = f"{label}-Relevance in [{args.min_relevance}, {args.max_relevance}] (Applicable not required)"
    print(f"\nNumber of samples with {summary}: {len(filtered_samples)}")

    if filtered_samples:
        source_dist_post = Counter(item["source"] for item in filtered_samples)
        print("\nSource distribution (post-filtering):")
        for source, count in sorted(source_dist_post.items(), key=lambda x: -x[1]):
            print(f"  {source}: {count:,} ({count / len(filtered_samples) * 100:.2f}%)")
        print(f"Total unique sources: {len(source_dist_post)}")
        print(f"Total filtered samples: {len(filtered_samples):,}")
    else:
        print("\nNo samples matched the filter criteria.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for item in filtered_samples:
            f.write(_json_dumps(item) + "\n")

    print(f"Saved filtered samples to: {output_path}")
