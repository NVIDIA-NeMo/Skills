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

"""Assemble the 30-problem development set used in the report from public Hugging Face datasets.

The set has 20 problems from Nemotron-IMO-Bench and 10 from recent competitions (IMO-ProofBench, EGMO via
MathNet, IMO 2025 via MathArena). The output is a pipeline input file: one row per problem with ``problem_idx``,
``question``, and ``source_name``, in the report's order (easy, medium, hard, unsolved tiers).

    python recipes/nemotron-imo-tts/scripts/build_devset30.py \\
        --nemotron-imo-bench <hugging-face-dataset-id-or-url> --output devset30.jsonl

    python recipes/nemotron-imo-tts/scripts/build_devset30.py --public-only --output devset30-public10.jsonl
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from pathlib import Path

HF_API = "https://datasets-server.huggingface.co"
SOURCE_NAME = "devset30"

# (tier, problem id) in the report's order. Ids without the proofbench prefix belong to Nemotron-IMO-Bench.
DEVSET30 = [
    ("EASY", "G50"),
    ("EASY", "A22"),
    ("EASY", "proofbench133_101"),
    ("MEDIUM", "A48"),
    ("MEDIUM", "G17"),
    ("MEDIUM", "A3"),
    ("MEDIUM", "C5"),
    ("MEDIUM", "N20"),
    ("MEDIUM", "proofbench133_093"),
    ("MEDIUM", "proofbench133_008"),
    ("HARD", "A32"),
    ("HARD", "C32"),
    ("HARD", "proofbench133_130"),
    ("HARD", "N30"),
    ("HARD", "N39"),
    ("HARD", "proofbench133_121"),
    ("HARD", "G43"),
    ("HARD", "N14"),
    ("HARD", "proofbench133_112"),
    ("HARD", "proofbench133_103"),
    ("UNSOLVED", "C19"),
    ("UNSOLVED", "N40"),
    ("UNSOLVED", "C13"),
    ("UNSOLVED", "G18"),
    ("UNSOLVED", "N42"),
    ("UNSOLVED", "G6"),
    ("UNSOLVED", "N18"),
    ("UNSOLVED", "proofbench133_118"),
    ("UNSOLVED", "proofbench133_120"),
    ("UNSOLVED", "proofbench133_133"),
]

BENCHMARK_IDS = {problem_id for _, problem_id in DEVSET30 if not problem_id.startswith("proofbench133_")}

# The ten public-competition problems and where each comes from.
PUBLIC_SOURCES = [
    {
        "dataset": "Hwilner/imo-proofbench",
        "config": "default",
        "split": "train",
        "id_field": "Problem ID",
        "problem_field": "Problem",
        "dataset_license": "CC-BY-4.0",
        "ids": {
            "PB-Advanced-004": "proofbench133_101",
            "PB-Basic-026": "proofbench133_093",
            "PB-Advanced-024": "proofbench133_121",
            "PB-Advanced-015": "proofbench133_112",
            "PB-Advanced-006": "proofbench133_103",
            "PB-Advanced-021": "proofbench133_118",
            "PB-Advanced-023": "proofbench133_120",
        },
    },
    {
        "dataset": "ShadenA/MathNet",
        "config": "European_Girls'_Mathematical_Olympiad_EGMO",
        "split": "train",
        "id_field": "id",
        "problem_field": "problem_markdown",
        "dataset_license": "CC-BY-4.0",
        "ids": {"05ed": "proofbench133_008"},
    },
    {
        "dataset": "MathArena/imo_2025",
        "config": "default",
        "split": "train",
        "id_field": "problem_idx",
        "problem_field": "problem",
        "dataset_license": "CC-BY-NC-SA-4.0",
        "ids": {"3": "proofbench133_130", "6": "proofbench133_133"},
    },
]

STATEMENT_FIELDS = ("problem", "statement", "problem_markdown", "question")
ID_FIELDS = ("id", "problem_idx", "problem_id")


def api_json(endpoint, **params):
    url = f"{HF_API}/{endpoint}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def download_rows(dataset, config, split):
    """Every row of one dataset split through the Hugging Face datasets server."""
    rows = []
    offset = 0
    while True:
        page = api_json("rows", dataset=dataset, config=config, split=split, offset=offset, length=100)
        rows.extend(item["row"] for item in page["rows"])
        if len(rows) >= page["num_rows_total"] or not page["rows"]:
            return rows
        offset = len(rows)


def parse_dataset_id(value):
    if "://" not in value:
        return value.strip("/")
    parsed = urllib.parse.urlparse(value)
    parts = parsed.path.strip("/").split("/")
    if parsed.netloc != "huggingface.co" or len(parts) < 3 or parts[0] != "datasets":
        raise ValueError(f"Not a Hugging Face dataset URL: {value}")
    return "/".join(parts[1:3])


def default_location(dataset, splits_api=api_json):
    locations = [(item["config"], item["split"]) for item in splits_api("splits", dataset=dataset)["splits"]]
    if ("default", "train") in locations:
        return "default", "train"
    if len(locations) == 1:
        return locations[0]
    raise ValueError(f"{dataset} has several configs/splits; pass a dataset with a default train split")


def collect_public(download=download_rows):
    problems = {}
    for source in PUBLIC_SOURCES:
        rows = download(source["dataset"], source["config"], source["split"])
        indexed = {str(row[source["id_field"]]): row for row in rows}
        missing = sorted(source["ids"].keys() - indexed.keys())
        if missing:
            raise ValueError(f"{source['dataset']} is missing source ids {missing}")
        for source_id, problem_id in source["ids"].items():
            statement = indexed[source_id][source["problem_field"]]
            if source["dataset"] == "ShadenA/MathNet":
                statement = statement.removeprefix("Problem:\n\n")
            problems[problem_id] = {
                "question": statement,
                "source_dataset": f"https://huggingface.co/datasets/{source['dataset']}",
                "source_id": source_id,
                "dataset_license": source["dataset_license"],
            }
    return problems


def collect_benchmark(dataset, rows):
    problems = {}
    for row in rows:
        source_id = next((str(row[field]) for field in ID_FIELDS if row.get(field) is not None), None)
        if source_id not in BENCHMARK_IDS:
            continue
        statement = next((row[field] for field in STATEMENT_FIELDS if row.get(field)), None)
        if statement is None:
            raise ValueError(f"benchmark row {source_id} has no statement field among {STATEMENT_FIELDS}")
        problems[source_id] = {
            "question": statement,
            "source_dataset": f"https://huggingface.co/datasets/{dataset}",
            "source_id": source_id,
            "dataset_license": "see the dataset card",
        }
    missing = sorted(BENCHMARK_IDS - problems.keys())
    if missing:
        raise ValueError(f"{dataset} is missing benchmark ids {missing}")
    return problems


def assemble(problems, public_only=False):
    rows = []
    for tier, problem_id in DEVSET30:
        if problem_id not in problems:
            if public_only and problem_id in BENCHMARK_IDS:
                continue
            raise ValueError(f"missing problem {problem_id}")
        entry = problems[problem_id]
        rows.append(
            {
                "problem_idx": problem_id,
                "question": entry["question"],
                "source_name": SOURCE_NAME,
                "tier": tier,
                "source_dataset": entry["source_dataset"],
                "source_id": entry["source_id"],
                "dataset_license": entry["dataset_license"],
            }
        )
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nemotron-imo-bench", help="Hugging Face dataset id or URL of Nemotron-IMO-Bench")
    parser.add_argument("--public-only", action="store_true", help="Write only the ten public-competition problems")
    parser.add_argument("--output", type=Path, default=Path("devset30.jsonl"))
    args = parser.parse_args(argv)
    if not args.public_only and not args.nemotron_imo_bench:
        parser.error("--nemotron-imo-bench is required unless --public-only is given")

    problems = collect_public()
    if args.nemotron_imo_bench:
        dataset = parse_dataset_id(args.nemotron_imo_bench)
        config, split = default_location(dataset)
        problems.update(collect_benchmark(dataset, download_rows(dataset, config, split)))
    rows = assemble(problems, public_only=args.public_only)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} problems to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
