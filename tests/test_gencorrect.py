# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from recipes.gencorrect import prepare_next_round


def evaluation_row(base_row: dict, score: float, generated_tokens: int) -> dict:
    code = "#include <bits/stdc++.h>\nint main(){" + "int value=0;value++;" * 20 + "return value;}"
    return {
        **base_row,
        "generation": f"```cpp\n{code}\n```",
        "num_generated_tokens": generated_tokens,
        "test_case_results": {
            "full": {
                "score": score,
                "max_score": 100.0,
                "outputs": [{"compile_success": True}],
            }
        },
    }


def test_scores_outside_shortlist_do_not_affect_selection() -> None:
    base_row = {"id": "demo", "problem_id": "demo", "subtask": "full", "problem": "Demo"}
    evaluations = [(rs, evaluation_row(base_row, 100.0 if rs == 10 else float(rs), 1_000 + rs)) for rs in range(11)]

    _, output_row = prepare_next_round.build_rows([base_row], evaluations)[0]

    assert output_row["gencorrect_submission_rs"] == list(range(10))
    assert output_row["gencorrect_rs"] == 9
    assert output_row["achieved_subtask_scores"] == {"full": 9.0}


def test_incomplete_generation_round_is_rejected(tmp_path) -> None:
    base_row = {"id": "demo", "problem_id": "demo", "subtask": "full"}
    (tmp_path / "output-rs0.jsonl").write_text(json.dumps(base_row) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Incomplete generation round"):
        prepare_next_round.load_evaluations(tmp_path, {prepare_next_round.row_key(base_row)}, num_runs=2)
