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

"""Deterministic work identities and the on-disk ledger that makes relaunches resume instead of redo.

Every request has an identity derived from what it computes. Its response is persisted as one JSONL row that
carries the identity's fields. On launch the ledger scans a problem's directory; any request whose identity is
already on disk is replayed from the row and never re-issued. A row with any finish reason is completed work;
only requests that produced no response (errors, parked) are missing.
"""

import re
from pathlib import Path

from nemotron_imo_tts.text import is_complete_finish_reason, load_jsonl, sha256_text

ROLE_FILES = {"gen": "gen.jsonl", "verify": "verify.jsonl", "refine": "refine.jsonl"}
JUDGE_FILE = Path("selection") / "judge.jsonl"
_ROUND_DIR_RE = re.compile(r"^R(\d+)$")


def gen_identity(problem_idx, prompt_id, model_id, slot):
    return f"gen:{problem_idx}:R1:{prompt_id}:{model_id}:{slot}"


def refine_identity(problem_idx, round_idx, trial_idx, model_id, slot):
    return f"refine:{problem_idx}:R{round_idx}:{trial_idx}:{model_id}:{slot}"


def verify_identity(problem_idx, round_idx, proof_sha, verifier_id, slot):
    return f"verify:{problem_idx}:R{round_idx}:{proof_sha}:{verifier_id}:{slot}"


def judge_identity(problem_idx, proof_sha, judge_id, slot, attempt):
    return f"judge:{problem_idx}:{proof_sha}:{judge_id}:{slot}:a{attempt}"


def row_proof_sha(row):
    sha = row.get("proof_sha256")
    if isinstance(sha, str) and sha:
        return sha
    proof = row.get("proof")
    if isinstance(proof, str):
        return sha256_text(proof)
    return None


def row_identity(row, role, round_idx):
    """The identity of a persisted response row, or None for rows that are not completed responses."""
    if not isinstance(row, dict) or "generation" not in row:
        return None
    problem_idx = row.get("problem_idx")
    if problem_idx is None:
        return None
    try:
        if role == "gen":
            return gen_identity(
                problem_idx,
                row["generation_prompt_id"],
                row["generation_model_id"],
                int(row["generation_random_seed"]),
            )
        if role == "refine":
            return refine_identity(
                problem_idx,
                round_idx,
                int(row["aggregation_trial_idx"]),
                row["generation_model_id"],
                int(row["generation_random_seed"]),
            )
        if role == "verify":
            sha = row_proof_sha(row)
            if sha is None:
                return None
            return verify_identity(problem_idx, round_idx, sha, row["verifier_id"], int(row["verification_seed"]))
        if role == "judge":
            sha = row_proof_sha(row)
            if sha is None:
                return None
            return judge_identity(problem_idx, sha, row["judge_id"], int(row["slot"]), int(row["attempt"]))
    except (KeyError, TypeError, ValueError):
        return None
    return None


class Ledger:
    """Completed work found under one problem directory. The first row for an identity wins."""

    def __init__(self):
        self._row_by_identity = {}
        self._rows_by_role_round = {}
        self._judge_attempts = {}

    @classmethod
    def scan(cls, problem_dir):
        ledger = cls()
        problem_dir = Path(problem_dir)
        rounds_dir = problem_dir / "rounds"
        if rounds_dir.is_dir():
            round_dirs = []
            for entry in rounds_dir.iterdir():
                match = _ROUND_DIR_RE.match(entry.name)
                if match and entry.is_dir():
                    round_dirs.append((int(match.group(1)), entry))
            for round_idx, round_dir in sorted(round_dirs):
                for role, file_name in ROLE_FILES.items():
                    path = round_dir / file_name
                    if path.is_file():
                        for row in load_jsonl(path):
                            ledger._add(row, role, round_idx)
        judge_path = problem_dir / JUDGE_FILE
        if judge_path.is_file():
            for row in load_jsonl(judge_path):
                ledger._add(row, "judge", 0)
        return ledger

    def _add(self, row, role, round_idx):
        identity = row_identity(row, role, round_idx)
        if identity is None:
            return
        existing = self._row_by_identity.get(identity)
        if existing is not None:
            # This pipeline never writes two rows for one identity. Directories produced by earlier pipeline
            # versions can, because they re-issued truncated responses; there the complete row is the one that
            # fed the run, so it supersedes a truncated one. Otherwise the first row wins.
            if is_complete_finish_reason(existing.get("finish_reason")) or not is_complete_finish_reason(
                row.get("finish_reason")
            ):
                return
            self._replace(existing, row, role, round_idx)
            return
        self._row_by_identity[identity] = row
        self._rows_by_role_round.setdefault((role, round_idx), []).append(row)
        if role == "judge":
            key = (row_proof_sha(row), row["judge_id"], int(row["slot"]))
            self._judge_attempts.setdefault(key, []).append(row)

    def _replace(self, existing, row, role, round_idx):
        identity = row_identity(row, role, round_idx)
        self._row_by_identity[identity] = row
        rows = self._rows_by_role_round.get((role, round_idx), [])
        for i, candidate in enumerate(rows):
            if candidate is existing:
                rows[i] = row
                break
        if role == "judge":
            key = (row_proof_sha(row), row["judge_id"], int(row["slot"]))
            attempts = self._judge_attempts.get(key, [])
            for i, candidate in enumerate(attempts):
                if candidate is existing:
                    attempts[i] = row
                    break

    def lookup(self, identity):
        return self._row_by_identity.get(identity)

    def rows(self, role, round_idx):
        return list(self._rows_by_role_round.get((role, round_idx), []))

    def judge_attempts(self, proof_sha, judge_id, slot):
        return sorted(self._judge_attempts.get((proof_sha, judge_id, slot), []), key=lambda r: int(r["attempt"]))

    def __len__(self):
        return len(self._row_by_identity)
