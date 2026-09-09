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

"""Text parsing shared by every stage: proof extraction, boxed scores, and small JSON/JSONL helpers."""

import hashlib
import json
import os
import re
import uuid
from pathlib import Path

UNFINISHED_PROOF = "UNFINISHED PROOF GENERATION"
NULL_SELF_EVAL = {"self_eval": "null", "self_eval_score": 0}


def strip_think(text):
    """Drop a reasoning prefix that some servers inline before the answer."""
    if text is None:
        return ""
    text = text.strip()
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


def extract_boxed_answers(text):
    """Return the contents of every ``\\boxed{...}`` in order, honoring nested braces."""
    answers = []
    for piece in text.split("boxed{")[1:]:
        depth = 0
        for i, ch in enumerate(piece):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    if i + 1 < len(piece) and piece[i + 1] == "%":
                        answers.append(piece[: i + 1])
                    else:
                        answers.append(piece[:i])
                    break
    return answers


def _normalize_prover_output(text):
    text = text.strip()
    text = re.sub(r"(^|\n)\s*\*+\s*Solution\s*\*+\s*\n", "\n## Solution\n", text)
    text = re.sub(r"\n\s*\*+\s*Self Evaluation\s*\*+\s*\n", "\n## Self Evaluation\n", text)
    text = re.sub(r"(^|\n)## Solution\s*\n", "\n## Solution\n", text)
    text = re.sub(r"\n## Self Evaluation\s*\n", "\n## Self Evaluation\n", text)
    return text.strip()


def extract_solution(text):
    """Text between ``## Solution`` and ``## Self Evaluation`` (audit sections stay inside the proof)."""
    text = _normalize_prover_output(text)
    return re.split(r"## Solution\s*\n", re.split(r"\n## Self Evaluation\s*\n", text)[0])[1].strip()


def extract_self_eval(text):
    text = _normalize_prover_output(text)
    return re.split(r"\n## Self Evaluation\s*\n", text)[1].strip()


def is_complete_finish_reason(finish_reason):
    if finish_reason is None:
        return True
    return str(finish_reason).lower() in ("stop", "eos", "eos_token")


def parse_proof(generation, finish_reason):
    """Split a generation into (proof, self-evaluation, valid).

    A truncated response is invalid. A response without the two sections keeps the whole text as the proof
    (valid when non-empty) with a null self-evaluation, matching the report run.
    """
    if not is_complete_finish_reason(finish_reason):
        return UNFINISHED_PROOF, dict(NULL_SELF_EVAL), False
    text = strip_think(generation)
    try:
        self_eval_text = extract_self_eval(text).strip()
        solution_text = extract_solution(text).strip()
    except Exception:
        return text.strip(), dict(NULL_SELF_EVAL), bool(text.strip())
    score = 0.0
    try:
        scores = [s.strip() for s in extract_boxed_answers(self_eval_text) if s.strip()]
        if scores:
            score = float(scores[-1])
    except Exception:
        score = 0.0
    return solution_text, {"self_eval": self_eval_text, "self_eval_score": score}, bool(solution_text)


def parse_verification_score(text):
    """The verifier's final boxed score: 0, 0.5, or 1; anything else is an invalid judgment."""
    if text is None:
        return None
    scores = [s.strip() for s in extract_boxed_answers(strip_think(text)) if s.strip()]
    if not scores:
        return None
    try:
        score = float(scores[-1])
    except ValueError:
        return None
    if score in (0.0, 0.5, 1.0):
        return score
    return None


def parse_judge_score(text):
    """The IMO-style judge's final boxed score: an integer from 0 to 7 (integral floats accepted)."""
    if text is None:
        return None
    scores = [s.strip() for s in extract_boxed_answers(strip_think(text)) if s.strip()]
    if not scores:
        return None
    try:
        value = float(scores[-1])
    except ValueError:
        return None
    if not value.is_integer() or not 0 <= value <= 7:
        return None
    return int(value)


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path, skip_bad_lines=True):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                if not skip_bad_lines:
                    raise ValueError(f"Invalid JSON at {path}:{line_num}") from None
    return rows


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
