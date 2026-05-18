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

import json

import pytest

from nemo_skills.adapters.gym_to_skills import (
    convert_file,
    convert_math_rollout,
    convert_rollouts,
    main,
    supported_metric_types,
)


# A reduced-but-realistic Gym math_with_judge rollout shape derived from
# Gym/resources_servers/math_with_judge/data/example_rollouts.jsonl
def _gym_rollout(
    *,
    library_reward=1.0,
    reward=1.0,
    extracted="42",
    expected="42",
    output_text=r"\boxed{42}",
    judge_evaluations=None,
):
    return {
        "reward": reward,
        "library_reward": library_reward,
        "extracted_answer": extracted,
        "expected_answer": expected,
        "judge_evaluations": judge_evaluations,
        "responses_create_params": {"input": [{"role": "user", "content": "..."}]},
        "response": {
            "output": [
                {
                    "type": "reasoning",
                    "summary": [],
                },
                {
                    "type": "message",
                    "content": [
                        {"type": "output_text", "text": output_text},
                    ],
                },
            ]
        },
    }


class TestConvertMathRollout:
    def test_correct_rollout_maps_to_symbolic_true(self):
        out = convert_math_rollout(_gym_rollout(library_reward=1.0))
        assert out["symbolic_correct"] is True
        assert out["predicted_answer"] == "42"
        assert out["expected_answer"] == "42"
        assert out["generation"] == r"\boxed{42}"

    def test_incorrect_rollout_maps_to_symbolic_false(self):
        out = convert_math_rollout(_gym_rollout(library_reward=0.0))
        assert out["symbolic_correct"] is False

    def test_missing_library_reward_falls_back_to_reward(self):
        # Some resource servers don't emit library_reward separately.
        rollout = _gym_rollout(library_reward=1.0, reward=0.0)
        del rollout["library_reward"]
        out = convert_math_rollout(rollout)
        # Falls back to reward (0.0) → False.
        assert out["symbolic_correct"] is False

    def test_concatenates_multiple_output_text_chunks(self):
        rollout = _gym_rollout()
        rollout["response"]["output"] = [
            {"type": "message", "content": [{"type": "output_text", "text": "part one "}]},
            {"type": "message", "content": [{"type": "output_text", "text": "part two"}]},
        ]
        out = convert_math_rollout(rollout)
        assert out["generation"] == "part one part two"

    def test_handles_empty_response(self):
        rollout = _gym_rollout()
        rollout["response"] = {}
        out = convert_math_rollout(rollout)
        assert out["generation"] == ""

    def test_judge_evaluations_surface_as_judgement_string(self):
        judge = [
            {
                "response": {
                    "output": [
                        {
                            "type": "message",
                            "content": [{"type": "output_text", "text": "Verdict: Yes, the answers match."}],
                        }
                    ]
                }
            }
        ]
        out = convert_math_rollout(_gym_rollout(judge_evaluations=judge))
        assert "judgement" in out
        assert "Yes" in out["judgement"]

    def test_no_judgement_field_when_judge_evaluations_is_null(self):
        out = convert_math_rollout(_gym_rollout(judge_evaluations=None))
        assert "judgement" not in out

    def test_expected_answer_passed_through_with_latex(self):
        # Realistic example: math_with_judge keeps LaTeX wrappers.
        out = convert_math_rollout(_gym_rollout(expected=r"\(\frac{333}{1997}\)"))
        assert out["expected_answer"] == r"\(\frac{333}{1997}\)"


class TestDispatch:
    def test_supported_metric_types_includes_math(self):
        assert "math" in supported_metric_types()

    def test_unsupported_metric_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="metric_type"):
            list(convert_rollouts([_gym_rollout()], metric_type="totally_made_up"))


class TestFileLevel:
    def test_convert_file_roundtrip(self, tmp_path):
        rollouts = [
            _gym_rollout(library_reward=1.0, extracted="1", expected="1"),
            _gym_rollout(library_reward=0.0, extracted="2", expected="3"),
        ]
        src = tmp_path / "rollouts.jsonl"
        dst = tmp_path / "output.jsonl"
        with src.open("w") as f:
            for r in rollouts:
                f.write(json.dumps(r) + "\n")

        count = convert_file(src, dst, metric_type="math")

        assert count == 2
        lines = [json.loads(line) for line in dst.read_text().splitlines()]
        assert lines[0]["symbolic_correct"] is True
        assert lines[0]["predicted_answer"] == "1"
        assert lines[1]["symbolic_correct"] is False
        assert lines[1]["expected_answer"] == "3"

    def test_convert_file_handles_blank_lines(self, tmp_path):
        src = tmp_path / "rollouts.jsonl"
        dst = tmp_path / "output.jsonl"
        src.write_text(json.dumps(_gym_rollout()) + "\n\n" + json.dumps(_gym_rollout()) + "\n")
        count = convert_file(src, dst, metric_type="math")
        assert count == 2

    def test_convert_file_creates_parent_dir(self, tmp_path):
        src = tmp_path / "rollouts.jsonl"
        dst = tmp_path / "nested" / "subdir" / "output.jsonl"
        src.write_text(json.dumps(_gym_rollout()) + "\n")
        convert_file(src, dst, metric_type="math")
        assert dst.exists()


class TestCLI:
    def test_cli_exits_zero_on_success(self, tmp_path):
        src = tmp_path / "rollouts.jsonl"
        dst = tmp_path / "output.jsonl"
        src.write_text(json.dumps(_gym_rollout()) + "\n")
        rc = main([str(src), str(dst), "--metric_type=math"])
        assert rc == 0
        assert dst.exists()

    def test_cli_rejects_unknown_metric_type(self, tmp_path, capsys):
        src = tmp_path / "rollouts.jsonl"
        src.write_text("")
        with pytest.raises(SystemExit):
            # argparse exits non-zero on bad choices.
            main([str(src), str(tmp_path / "out.jsonl"), "--metric_type=foo"])
