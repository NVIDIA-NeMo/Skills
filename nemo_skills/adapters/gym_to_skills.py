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
"""Convert Gym `ng_collect_rollouts` output into a Skills-shape `output.jsonl`.

The existing `nemo_skills.pipeline.summarize_results` and the
`nemo_skills.evaluation.metrics.MathMetrics` calculator read fields like
`symbolic_correct`, `predicted_answer`, `expected_answer`, and `generation`
from Skills' `output.jsonl`. Rather than teach `summarize_results` about
Gym's `rollouts.jsonl` schema, we write a parallel Skills-shape file
alongside each Gym rollouts file so the downstream metric path is unchanged.

For the v1 pilot only `math` is supported (the `math_with_judge` resource
server). Other `metric_type`s raise — when we expand benchmarks in Tier 2/3
we add per-type converters here.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

LOG = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# math: math_with_judge → Skills MathMetrics
# ----------------------------------------------------------------------


def _extract_output_text(response: Dict[str, Any]) -> str:
    """Pull the assistant-message text out of an OpenAI Responses-API response."""
    chunks: List[str] = []
    for item in response.get("output", []) or []:
        for c in item.get("content", []) or []:
            if c.get("type") == "output_text":
                text = c.get("text")
                if text:
                    chunks.append(text)
    return "".join(chunks)


def convert_math_rollout(rollout: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one Gym `math_with_judge` rollout dict into a Skills math prediction.

    Maps the fields that `MathMetrics` reads:
        symbolic_correct ← (library_reward == 1.0)
        predicted_answer ← extracted_answer
        expected_answer  ← expected_answer
        generation       ← response.output[].content[].text (concatenated)

    If a judge produced evaluations, surface them as `judgement` so Skills'
    `is_correct_judgement` can parse them. v1 pilot benchmarks (gsm8k) don't
    use the judge, so this is a best-effort placeholder.
    """
    library_reward = rollout.get("library_reward")
    if library_reward is None:
        # Fall back to the overall reward when the server doesn't expose the
        # symbolic-only signal (we lose the judge/symbolic separation but at
        # least we get a pass/fail).
        library_reward = rollout.get("reward", 0.0)

    out: Dict[str, Any] = {
        "generation": _extract_output_text(rollout.get("response", {}) or {}),
        "predicted_answer": rollout.get("extracted_answer"),
        "expected_answer": rollout.get("expected_answer"),
        "symbolic_correct": bool(library_reward == 1.0),
    }

    judge = rollout.get("judge_evaluations")
    if judge:
        # MathMetrics calls is_correct_judgement(prediction["judgement"]) which
        # expects a string. The judge_evaluations entries each have their own
        # `response.output[].content[].text`; concatenate the first verdict
        # text for now and revisit when Tier 2 brings judge benchmarks in.
        verdicts = []
        for ev in judge if isinstance(judge, list) else [judge]:
            response = ev.get("response", {}) if isinstance(ev, dict) else {}
            text = _extract_output_text(response)
            if text:
                verdicts.append(text)
        if verdicts:
            out["judgement"] = "\n\n".join(verdicts)

    return out


# ----------------------------------------------------------------------
# Dispatch + file IO
# ----------------------------------------------------------------------


_CONVERTERS = {
    "math": convert_math_rollout,
}


def supported_metric_types() -> List[str]:
    return sorted(_CONVERTERS.keys())


def convert_rollouts(
    rollouts: Iterable[Dict[str, Any]],
    *,
    metric_type: str,
) -> Iterable[Dict[str, Any]]:
    try:
        converter = _CONVERTERS[metric_type]
    except KeyError as e:
        raise NotImplementedError(
            f"gym_to_skills: no converter for metric_type={metric_type!r}. "
            f"Supported: {supported_metric_types()}. "
            f"Add a converter to nemo_skills/adapters/gym_to_skills.py."
        ) from e
    for rollout in rollouts:
        yield converter(rollout)


def convert_file(
    rollouts_path: str | Path,
    output_path: str | Path,
    *,
    metric_type: str,
) -> int:
    """Read `rollouts_path`, write Skills-shape JSONL to `output_path`. Returns row count."""
    rollouts_path = Path(rollouts_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with rollouts_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        rollouts = (json.loads(line) for line in fin if line.strip())
        for prediction in convert_rollouts(rollouts, metric_type=metric_type):
            fout.write(json.dumps(prediction))
            fout.write("\n")
            n += 1
    LOG.info("Wrote %d converted rows from %s to %s", n, rollouts_path, output_path)
    return n


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m nemo_skills.adapters.gym_to_skills",
        description="Convert Gym rollouts.jsonl into Skills-shape output.jsonl.",
    )
    p.add_argument("rollouts", help="Path to the Gym rollouts JSONL file.")
    p.add_argument("output", help="Path to write the Skills-shape JSONL file.")
    p.add_argument(
        "--metric_type",
        required=True,
        choices=supported_metric_types(),
        help="Which Skills metric calculator the output will feed.",
    )
    return p


def main(argv: List[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _build_argparser().parse_args(argv)
    convert_file(args.rollouts, args.output, metric_type=args.metric_type)
    return 0


if __name__ == "__main__":
    sys.exit(main())
