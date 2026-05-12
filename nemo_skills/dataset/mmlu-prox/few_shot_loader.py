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

"""Load first n examples from MMLU-ProX validation split per language for few-shot prompting."""

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import load_dataset

# Load prepare from same directory (folder name has hyphen so package import is not used)
_prepare_spec = importlib.util.spec_from_file_location(
    "_mmlu_prox_prepare", Path(__file__).resolve().parent / "prepare.py"
)
_prepare = importlib.util.module_from_spec(_prepare_spec)
_prepare_spec.loader.exec_module(_prepare)
download_and_parse_lang_libs = _prepare.download_and_parse_lang_libs


# Cache lang_libs and per-language validation examples (mutate in place to avoid global declaration)
_lang_libs_cache = [None]  # [0] = (lang_libs, lang_subjects)
_validation_examples_cache = {}


def _format_validation_row_to_few_shot(entry: Dict[str, Any], language: str, lang_libs: Dict[str, Any]) -> dict:
    """Format a single MMLU-ProX validation row to {problem, solution, topic} for the few-shot template.

    Matches lm-evaluation-harness format_cot_example(): problem is [0] + question + [1] + options;
    solution is cot_content with lang_libs[4] replaced by lang_libs[2], or the answer phrase [5] if no cot.
    """
    category = entry["category"].replace(" ", "_")
    ld = lang_libs[language]
    # Problem text matches test rows from prepare.py:
    # question marker + question + options marker + options + answer prompt.
    problem = f"{ld[0]}\n{entry['question']}\n{ld[1]}\n"
    for i in range(10):
        opt = entry[f"option_{i}"]
        if opt is not None:
            problem += f"{chr(ord('A') + i)}. {opt}\n"
    problem += f"{ld[2]}\n"
    # Solution: keep exactly one answer-prefix line per few-shot example.
    # The problem block already ends with ld[2], so strip any leading
    # answer prefix from cot_content to avoid duplicates.
    cot_content = entry.get("cot_content") or ""
    if cot_content:
        solution = cot_content.lstrip()
        for prefix in (ld[4], ld[2]):
            if solution.startswith(prefix):
                solution = solution[len(prefix) :].lstrip()
        solution = solution + "\n\n"
    else:
        solution = ld[5].format(entry["answer"])
    return {"problem": problem, "solution": solution, "topic": category}


def load_mmlu_prox_validation_few_shot(
    language: str,
    n: int = 5,
    lang_libs: Optional[Dict[str, Any]] = None,
    lang_subjects: Optional[Dict[str, Any]] = None,
) -> List[dict]:
    """
    Load the first n examples from the MMLU-ProX validation split for the given language.

    Uses the same formatting as the prepared dataset (lang_libs, subject descriptions)
    so few-shot examples match the test prompt style. Each example has keys:
    problem, solution, topic (for compatibility with few-shot template).

    Args:
        language: Language code (e.g. "en", "de", "ja") as in li-lab/MMLU-ProX.
        n: Number of validation examples to take from the start (default 5).
        lang_libs: Optional pre-loaded LANG_LIBS (avoids re-download if provided).
        lang_subjects: Optional pre-loaded LANG_SUBJECTS.

    Returns:
        List of dicts with keys "problem", "solution", "topic".
    """
    cache_key = (language, n)
    if cache_key in _validation_examples_cache:
        return _validation_examples_cache[cache_key]

    if lang_libs is None or lang_subjects is None:
        if _lang_libs_cache[0] is None:
            _lang_libs_cache[0] = download_and_parse_lang_libs()
        lang_libs, lang_subjects = _lang_libs_cache[0]

    if language not in lang_subjects:
        raise ValueError(f"Language {language} not found in MMLU-ProX validation split")

    ds = load_dataset("li-lab/MMLU-ProX", language, split="validation", trust_remote_code=True)

    examples = []
    for i in range(min(n, len(ds))):
        row = ds[i]
        examples.append(_format_validation_row_to_few_shot(row, language, lang_libs))

    _validation_examples_cache[cache_key] = examples
    return examples
