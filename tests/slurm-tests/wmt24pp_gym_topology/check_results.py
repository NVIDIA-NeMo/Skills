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
"""Check that every wmt24pp_gym_topology config produced per-row comet_score.

Invariant under test (from streaming actor pool + awaited dispatch in
verify()):

    rollout.comet_score is None  ⟺  rollout.generation == ""

i.e., empty-generation rows skip COMET (verify() early-returns), every
other row gets a score. We assert:

  1. Each topology's rollouts.jsonl exists and is non-empty.
  2. Total row count is at least the smallest expected limit (250 by default,
     but accept anything ≥ 200 to tolerate the rare retry-induced row).
  3. For every row with a non-empty ``generation``, ``comet_score`` is a finite
     float in [-1.5, 1.5] (xCOMET reports raw and normalized — both fall
     inside that band).
  4. Every row with ``comet_score is None`` has an empty ``generation``.

Failures are reported via the shared soft_assert helper so all 5 topologies
get checked even if one fails; the script exits non-zero only at the end
if any check failed.
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # for utils.py
from utils import assert_all, soft_assert  # noqa: E402

CONFIGS = ["c1", "c2", "c3", "c4", "c5"]
MIN_ROWS = 200
COMET_LO, COMET_HI = -1.5, 1.5


def _load_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_topology(workspace: str, name: str) -> None:
    rollouts_path = os.path.join(workspace, name, "rollouts.jsonl")
    if not os.path.exists(rollouts_path):
        soft_assert(False, f"[{name}] missing rollouts.jsonl at {rollouts_path}")
        return
    rows = _load_rows(rollouts_path)
    soft_assert(len(rows) >= MIN_ROWS, f"[{name}] only {len(rows)} rows in rollouts.jsonl (expected ≥ {MIN_ROWS})")

    invariant_violations = []
    bad_score_rows = []
    scored_rows = 0

    for i, row in enumerate(rows):
        gen = row.get("generation") or ""
        score = row.get("comet_score")
        if gen == "":
            # Empty generation must NOT carry a comet_score (verify() early-return).
            if score is not None:
                invariant_violations.append(f"row {i}: empty generation but comet_score={score} (expected None)")
        else:
            # Non-empty generation MUST carry a finite numeric comet_score.
            if score is None:
                invariant_violations.append(f"row {i}: non-empty generation but comet_score=None")
                continue
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                bad_score_rows.append(f"row {i}: non-finite comet_score={score!r}")
                continue
            if not (COMET_LO <= score <= COMET_HI):
                bad_score_rows.append(f"row {i}: comet_score={score} out of band [{COMET_LO}, {COMET_HI}]")
                continue
            scored_rows += 1

    soft_assert(
        not invariant_violations,
        f"[{name}] {len(invariant_violations)} invariant violation(s); first 3: {invariant_violations[:3]}",
    )
    soft_assert(
        not bad_score_rows,
        f"[{name}] {len(bad_score_rows)} out-of-band score(s); first 3: {bad_score_rows[:3]}",
    )
    # Sanity: at least some rows should be scored. A run where every row had an
    # empty generation indicates the model collapsed, not a comet-pipeline win.
    soft_assert(
        scored_rows >= MIN_ROWS // 2,
        f"[{name}] only {scored_rows} rows had a comet_score (model may have collapsed)",
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, help="Workspace dir holding c1..c5 subdirs from run_test.py")
    args = ap.parse_args()

    for name in CONFIGS:
        check_topology(args.workspace, name)

    assert_all()


if __name__ == "__main__":
    main()
