# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
import urllib.parse
import urllib.request
from pathlib import Path

from tqdm import tqdm

from nemo_skills.dataset.utils import get_mcq_fields

# Matches the '정답: X' answer line the prompt instructs the model to produce.
# CLIcK mixes 4-choice and 5-choice items; the 5-choice prompt covers both.
_EXTRACT_REGEX = r"(?i)정답\s*[:：]\s*\**\(?([A-E])\)?\**"

# The CLIcK HF mirror is a single flat split with no category metadata, so we
# pull the per-subcategory JSON files directly from the canonical GitHub repo
# (https://github.com/rladmstn1714/CLIcK) to recover the labels used in the
# LREC-COLING 2024 paper. Each subcategory is split across one or more files
# named after the source exam (KIIP, Kedu, CSAT, TOPIK, KHB, PSE, PSAT).
_GITHUB_BASE = "https://raw.githubusercontent.com/rladmstn1714/CLIcK/main/Dataset"
_SUBDIR = {
    "Korean Economy": "Culture/Korean Economy",
    "Korean Geography": "Culture/Korean Geography",
    "Korean History": "Culture/Korean History",
    "Korean Law": "Culture/Korean Law",
    "Korean Politics": "Culture/Korean Politics",
    "Korean Popular": "Culture/Korean Popular",
    "Korean Society": "Culture/Korean Society",
    "Korean Tradition": "Culture/Korean Tradition",
    "Functional": "Language/Functional",
    "Grammar": "Language/Grammar",
    "Textual": "Language/Textual",
}
_FILES = {
    "Korean Economy": ["Economy_KIIP.json", "Economy_Kedu.json"],
    "Korean Geography": ["Geography_CSAT.json", "Geography_KIIP.json", "Geography_Kedu.json"],
    "Korean History": ["History_KHB.json", "History_Kedu.json", "History_PSE.json"],
    "Korean Law": ["Law_KIIP.json", "Law_PSAT.json"],
    "Korean Politics": ["Politics_KIIP.json", "Politics_Kedu.json"],
    "Korean Popular": ["Popular_KIIP.json", "Popular_Kedu.json"],
    "Korean Society": ["Society_KIIP.json", "Society_Kedu.json"],
    "Korean Tradition": ["Tradition_KIIP.json", "Tradition_Kedu.json"],
    "Functional": ["Functional_CSAT.json", "Functional_Kedu.json", "Functional_PSE.json"],
    "Grammar": ["Grammar_CSAT.json", "Grammar_Kedu.json", "Grammar_TOPIK.json"],
    "Textual": ["Textual_CSAT.json", "Textual_TOPIK.json"],
}


def _fetch(subcategory: str, filename: str):
    url = f"{_GITHUB_BASE}/{urllib.parse.quote(_SUBDIR[subcategory])}/{filename}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read())


def format_entry(entry, subcategory: str):
    # The HF / GitHub schema stores the answer as the literal text of the
    # correct choice; map it back to the corresponding letter so the
    # per-sample regex can match.
    answer_index = entry["choices"].index(entry["answer"])
    expected_letter = chr(ord("A") + answer_index)

    # Some items carry an extra `paragraph` context that the question
    # refers to; prepend it when present.
    paragraph = entry.get("paragraph", "").strip()
    question = entry["question"].strip()
    if paragraph:
        question = f"{paragraph}\n\n{question}"

    return {
        "expected_answer": expected_letter,
        "extract_from_boxed": False,
        "extract_regex": _EXTRACT_REGEX,
        "relaxed": False,
        "subset_for_metrics": subcategory,
        "id": entry["id"],
        **get_mcq_fields(question, entry["choices"]),
    }


def main(args):
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / f"{args.split}.jsonl"

    rows = []
    for subcat, files in tqdm(_FILES.items(), desc="Fetching CLIcK"):
        for filename in files:
            for entry in _fetch(subcat, filename):
                rows.append(format_entry(entry, subcat))

    with open(output_file, "wt", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} items to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("test",), help="Dataset split to process.")
    args = parser.parse_args()
    main(args)
