# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use it except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared HotpotQA data formatting and preparation.

Used by both hotpotqa and hotpotqa_closedbook so there is a single source of truth
for downloading and formatting the distractor validation set.
"""

import json
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def format_context(context: dict) -> str:
    """Format context paragraphs with titles and indexed sentences.

    Each paragraph becomes:
        Title: <title>
        [0] <sentence 0>
        [1] <sentence 1>
        ...

    Paragraphs are separated by blank lines.
    """
    paragraphs = []
    for title, sentences in zip(context["title"], context["sentences"]):
        lines = [f"Title: {title}"]
        for idx, sent in enumerate(sentences):
            lines.append(f"[{idx}] {sent.strip()}")
        paragraphs.append("\n".join(lines))
    return "\n\n".join(paragraphs)


def format_entry(entry: dict) -> dict:
    """Format a HotpotQA entry to match NeMo-Skills format."""
    supporting_facts = list(zip(entry["supporting_facts"]["title"], entry["supporting_facts"]["sent_id"]))

    return {
        "id": entry["id"],
        "question": entry["question"],
        "expected_answer": entry["answer"],
        "context": format_context(entry["context"]),
        "supporting_facts": supporting_facts,
        "type": entry["type"],
        "level": entry["level"],
    }


def prepare_validation(output_path: Path) -> int:
    """Download HotpotQA distractor validation set and write NeMo-Skills format to output_path.

    Returns the number of examples written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

    with open(output_path, "wt", encoding="utf-8") as fout:
        for entry in tqdm(ds, desc=f"Writing {output_path.name}"):
            formatted = format_entry(entry)
            json.dump(formatted, fout)
            fout.write("\n")

    print(f"Wrote {len(ds)} examples to {output_path}")
    return len(ds)
