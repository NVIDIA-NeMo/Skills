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

# HotpotQA answer normalization and filtering.
# Adapted from hmaron/nvidia-research-tlv-nemotron-hallucination-detection,
# experiments/hotpotqa_filtering/eval_package/hotpotqa_eval.py
#
# Generates alternative surface forms of ground-truth answers and flags
# questions that are unreliable for substring-based evaluation.

import re

__all__ = [
    "is_correct",
    "is_correct_strict",
    "normalize_gt",
]

_NUM_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "first": "1st",
    "second": "2nd",
    "third": "3rd",
    "fourth": "4th",
    "fifth": "5th",
    "nineteenth": "19th",
    "twentieth": "20th",
    "twenty-first": "21st",
}
_NUM_DIGITS = {v: k for k, v in _NUM_WORDS.items()}

_MAX_GT_LENGTH = 40
_MIN_ALT_LENGTH = 3

_STOPWORDS = frozenset(
    [
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "for",
        "and",
        "or",
        "to",
        "by",
        "is",
        "was",
        "are",
        "were",
        "be",
        "been",
        "with",
        "from",
        "that",
        "this",
        "it",
        "its",
        "his",
        "her",
        "my",
        "our",
        "their",
        "no",
        "not",
        "but",
        "if",
        "as",
        "into",
        "about",
        "than",
        "then",
    ]
)


def _normalize_unicode(s: str) -> str:
    """Normalize unicode whitespace, hyphens, and quotes for matching."""
    for c in "\u202f\u00a0\u2009\u200a\u2002\u2003":
        s = s.replace(c, " ")
    for c in "\u2010\u2011\u2012\u2013\u2014\u2015":
        s = s.replace(c, "-")
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    while "  " in s:
        s = s.replace("  ", " ")
    return s.strip()


def _gt_alternatives(gt: str) -> tuple[list[str], list[str]]:
    """Generate valid surface-form alternatives for a ground-truth answer.

    Returns (sorted_alternatives, list_of_rule_tags_that_fired).
    """
    alts = {gt}
    rules = []

    for prefix in ("the ", "a ", "an "):
        if gt.lower().startswith(prefix):
            alts.add(gt[len(prefix) :])
            rules.append("strip_article")
            break

    stripped = gt.replace('"', "").replace("\u201c", "").replace("\u201d", "").strip()
    if stripped and stripped != gt:
        alts.add(stripped)
        rules.append("strip_quotes")

    if "(" in gt:
        no_parens = re.sub(r"\s*\([^)]*\)\s*", " ", gt).strip()
        no_parens = re.sub(r"\s+", " ", no_parens)
        if no_parens and no_parens != gt:
            alts.add(no_parens)
        for inner in re.findall(r"\(([^)]+)\)", gt):
            inner = inner.strip()
            if len(inner) > 1:
                alts.add(inner)
        rules.append("normalize_parens")

    gt_low = gt.lower().strip()
    if gt_low in _NUM_WORDS:
        alts.add(_NUM_WORDS[gt_low])
        rules.append("number_word_to_digit")
    if gt_low in _NUM_DIGITS:
        alts.add(_NUM_DIGITS[gt_low])
        rules.append("number_digit_to_word")

    no_commas = re.sub(r"(\d),(\d{3})", r"\1\2", gt)
    while no_commas != gt and re.search(r"(\d),(\d{3})", no_commas):
        no_commas = re.sub(r"(\d),(\d{3})", r"\1\2", no_commas)
    if no_commas != gt:
        alts.add(no_commas)
        rules.append("strip_number_commas")

    if gt and gt[-1] in ".,;:!?":
        alts.add(gt[:-1].rstrip())
        rules.append("strip_trailing_punct")

    if "." in gt:
        no_dots = re.sub(r"(?<!\d)\.(?!\d)", "", gt)
        if no_dots and len(no_dots) > 1 and no_dots != gt:
            alts.add(no_dots)
            rules.append("strip_abbrev_dots")

    if "-" in gt and not gt.startswith("-"):
        no_hyphen = re.sub(r"\s+", " ", gt.replace("-", " ")).strip()
        if no_hyphen != gt:
            alts.add(no_hyphen)
            rules.append("hyphen_to_space")

    if " & " in gt:
        alts.add(gt.replace(" & ", " and "))
        rules.append("ampersand_to_and")
    if " and " in gt.lower():
        idx = gt.lower().index(" and ")
        alts.add(gt[:idx] + " & " + gt[idx + 5 :])
        rules.append("and_to_ampersand")

    extra = set()
    for alt in list(alts):
        for prefix in ("the ", "a ", "an "):
            if alt.lower().startswith(prefix):
                extra.add(alt[len(prefix) :])
    alts |= extra

    normed = set()
    for a in alts:
        a = re.sub(r"\s+", " ", a.strip())
        if a and (len(a) >= _MIN_ALT_LENGTH or a == gt.strip() or a.isdigit()):
            normed.add(a)

    return sorted(normed), rules


def _is_multi_word_name(gt: str) -> bool:
    """True if GT looks like a multi-word proper name unreliable for substring matching."""
    parts = gt.strip().rstrip(".").split()
    n = len(parts)
    if n in (3, 4):
        return all(p[0].isupper() for p in parts) and all(p.lower() not in _STOPWORDS for p in parts)
    if n in (5, 6):
        caps = [p for p in parts if p[0].isupper() and p.lower() not in _STOPWORDS]
        return len(caps) >= 3
    return False


def _should_remove(gt: str) -> tuple[bool, str]:
    """Return (flag, reason). Reason is '' if not removed."""
    if len(gt) > _MAX_GT_LENGTH:
        return True, "gt_too_long"
    if _is_multi_word_name(gt):
        return True, "multi_word_name"
    return False, ""


def normalize_gt(gt_answer: str) -> dict:
    """Normalize a single ground-truth answer on-the-fly.

    Returns dict with keys:
        alternatives (list[str]): Valid surface forms (always includes original).
        should_remove (bool): True if unreliable for substring eval.
        remove_reason (str): '' | 'gt_too_long' | 'multi_word_name'.
        edited (bool): True if any rule fired.
        edit_reasons (list[str]): Tags of rules that fired.
    """
    alts, alt_rules = _gt_alternatives(gt_answer)
    remove, remove_reason = _should_remove(gt_answer)
    edit_reasons = list(alt_rules)
    if remove_reason:
        edit_reasons.append(remove_reason)
    return {
        "alternatives": alts,
        "should_remove": remove,
        "remove_reason": remove_reason,
        "edited": bool(edit_reasons),
        "edit_reasons": edit_reasons,
    }


def is_correct(alternatives: list[str], model_answer: str) -> bool:
    """Check if any alternative is a substring of the model answer.

    Args:
        alternatives: List from normalize_gt()['alternatives'].
        model_answer: The model's predicted answer string.
    """
    ans = _normalize_unicode(model_answer.lower())
    return any(_normalize_unicode(alt.lower()) in ans for alt in alternatives)


def is_correct_strict(alternatives: list[str], model_answer: str) -> bool:
    """Stricter matching that reduces false positives.

    Additional gates over is_correct():
      - Short alternatives (<=4 chars): require word-boundary match
      - Long model answers (>80 chars): reject if match starts after position 40
    """
    ans = _normalize_unicode(model_answer.lower())
    ans_len = len(ans)

    for alt in alternatives:
        alt_norm = _normalize_unicode(alt.lower())
        if not alt_norm:
            continue
        if alt_norm not in ans:
            continue
        if len(alt_norm) <= 4:
            if not re.search(r"(?<!\w)" + re.escape(alt_norm) + r"(?!\w)", ans):
                continue
        if ans_len > 80:
            pos = ans.find(alt_norm)
            if pos > 40:
                continue
        return True
    return False
