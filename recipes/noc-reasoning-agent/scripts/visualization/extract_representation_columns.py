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
import re
from collections import defaultdict

import pandas as pd

# ---------- Paths ----------
jsonl_path = "outputs/filtering_soft_with_keywords/output.jsonl"
output_jsonl_path = "outputs/filtering_soft_with_keywords/output_with_categories.jsonl"
csv_path = "data/human_intervention_incidents_soft.csv"
samples_dir = "outputs/filtering_soft_with_keywords/samples"

os.makedirs(os.path.dirname(output_jsonl_path), exist_ok=True)
os.makedirs(samples_dir, exist_ok=True)

# ---------- Load CSV (index by incident id: synthetic=incident_identifier, legacy=number) ----------
sample = pd.read_csv(csv_path, nrows=0)
id_col = "incident_identifier" if "incident_identifier" in sample.columns else "number"
df = pd.read_csv(csv_path, encoding="latin1", dtype={id_col: str})
df[id_col] = df[id_col].astype(str)

needed_cols = [id_col, "time_to_resolve", "solved_category", "solved_reason"]
missing = [c for c in needed_cols if c not in df.columns]
if missing:
    raise ValueError(f"CSV is missing required columns: {missing}")

csv_idx = df.set_index(id_col)[["time_to_resolve", "solved_category", "solved_reason"]].to_dict(orient="index")


# ---------- Helper: safe enrichment ----------
def enrich_row(row, lookup):
    num = row.get("incident_identifier", row.get("number"))
    info = lookup.get(num)
    if info is None:
        # No match: set to None (or choose sensible defaults)
        row["time_to_resolve"] = None
        row["solved_category"] = row.get("solved_category")  # preserve if already present
        row["solved_reason"] = row.get("solved_reason")
    else:
        row["time_to_resolve"] = info.get("time_to_resolve")
        row["solved_category"] = info.get("solved_category")
        row["solved_reason"] = info.get("solved_reason")
    return row


# ---------- Read input JSONL and enrich ----------
enriched_rows = []
with open(jsonl_path, "r", encoding="utf-8") as fin:
    for line in fin:
        if not line.strip():
            continue
        row = json.loads(line)
        row = enrich_row(row, csv_idx)
        enriched_rows.append(row)

# ---------- Write output JSONL ----------
with open(output_jsonl_path, "w", encoding="utf-8") as fout:
    for row in enriched_rows:
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"Wrote enriched JSONL → {output_jsonl_path}  (n={len(enriched_rows)})")

# ---------- Bucket by solved_category ----------
by_cat = defaultdict(list)
for r in enriched_rows:
    cat = r.get("resolution_method") or r.get("close_code") or "Unknown"
    by_cat[cat].append(r)


# ---------- Length bucketing rules (by token-ish count) ----------
def token_count(text: str) -> int:
    if not isinstance(text, str):
        return 0
    # crude token proxy: whitespace-split
    return len(text.split())


# You can tweak these thresholds if your generations are generally longer/shorter
SHORT_MAX = 1000  # tokens
MEDIUM_MAX = 5000  # tokens
# long: > MEDIUM_MAX


def length_bucket(text: str) -> str:
    n = token_count(text)
    if n <= SHORT_MAX:
        return "short"
    elif n <= MEDIUM_MAX:
        return "medium"
    return "long"


# ---------- Pick 5 samples per category with ≥1 short, ≥1 medium, ≥1 long (if available) ----------
def pick_samples(rows, n=5):
    # Build buckets
    buckets = {"short": [], "medium": [], "long": []}
    for r in rows:
        gen = r.get("generation", "")
        b = length_bucket(gen)
        buckets[b].append(r)

    # deterministic sort within buckets: by token length
    for b in buckets:
        buckets[b].sort(key=lambda x: token_count(x.get("generation", "")))

    picked = []

    # 1) ensure coverage: pick shortest short, median medium, longest long when available
    if buckets["short"]:
        picked.append(buckets["short"][0])  # shortest short
        buckets["short"] = buckets["short"][1:]

    if buckets["medium"]:
        mid = len(buckets["medium"]) // 2
        picked.append(buckets["medium"][mid])  # median medium
        buckets["medium"].pop(mid)

    if buckets["long"]:
        picked.append(buckets["long"][-1])  # longest long
        buckets["long"].pop(-1)

    # 2) fill remaining slots from the buckets in round-robin: short → medium → long
    order = ["short", "medium", "long"]
    i = 0
    while len(picked) < n and any(buckets[b] for b in order):
        b = order[i % 3]
        if buckets[b]:
            picked.append(buckets[b].pop(0))
        i += 1

    # If still short, just top up from whatever remains (unlikely)
    if len(picked) < n:
        remaining = buckets["short"] + buckets["medium"] + buckets["long"]
        picked.extend(remaining[: (n - len(picked))])

    return picked[:n]


# ---------- Write sample files per category ----------
def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


for cat, rows in by_cat.items():
    samples = pick_samples(rows, n=5)
    out_path = os.path.join(samples_dir, f"samples_{sanitize(cat)}.jsonl")
    with open(out_path, "w", encoding="utf-8") as fout:
        for r in samples:
            # Keep the whole record; downstream you can read r["generation"] for the trace
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    # quick summary
    lengths = [length_bucket(r.get("generation", "")) for r in samples]
    print(
        f"Category: {cat:>20} | total={len(rows):4d} | wrote={len(samples):2d} | mix={dict((x, lengths.count(x)) for x in set(lengths))} -> {out_path}"
    )
