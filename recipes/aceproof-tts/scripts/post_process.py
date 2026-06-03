#!/usr/bin/env python3

import argparse
import json
from copy import deepcopy
from pathlib import Path


def get_count(score_counts: dict, score: int) -> int:
    return int(score_counts.get(str(score), score_counts.get(score, 0)))


def postprocess_score(problem: dict) -> float:
    if get_count(problem["score_counts"], 0) > 0:
        return 0.0
    return float(problem["mean_score"])


def problem_category(problem_idx: str) -> str:
    if "Advanced" in problem_idx:
        return "advanced"
    if "Basic" in problem_idx:
        return "basic"
    raise ValueError(f"Cannot infer category from problem_idx={problem_idx}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Post-process metrics.json using the rule: if score_counts contains any "
            "0 score, processed_score=0; otherwise processed_score is mean_score."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to metrics.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to write processed metrics. Defaults to input sibling file.",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or input_path.with_name("metrics-processed-zero-else-mean.json")

    metrics = json.loads(input_path.read_text(encoding="utf-8"))
    output = deepcopy(metrics)

    output["postprocess_rule"] = {
        "name": "zero_if_any_zero_else_mean",
        "description": (
            "If score_counts contains any 0 score, processed_score is 0. Otherwise processed_score is mean_score."
        ),
    }

    category_scores = {name: [] for name in output.get("categories", {})}
    processed_problems = []

    for problem in output["problems"]:
        processed_problem = deepcopy(problem)
        processed = postprocess_score(processed_problem)
        category = problem_category(processed_problem["problem_idx"])

        processed_problem["processed_score"] = processed
        processed_problem["processed_score_pct"] = processed / 7.0
        processed_problem["processed_category"] = category

        category_scores.setdefault(category, []).append(processed)
        processed_problems.append(processed_problem)

    output["problems"] = processed_problems

    for category, summary in output.get("categories", {}).items():
        scores = category_scores.get(category, [])
        summary["processed_mean_score"] = sum(scores) / len(scores) if scores else None
        summary["processed_mean_score_pct"] = (
            summary["processed_mean_score"] / 7.0 if summary["processed_mean_score"] is not None else None
        )

    all_scores = [problem["processed_score"] for problem in output["problems"]]
    output["overall"]["processed_mean_score"] = sum(all_scores) / len(all_scores)
    output["overall"]["processed_mean_score_per_problem"] = output["overall"]["processed_mean_score"]

    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
