import argparse
import os
import re
import sys

PIPELINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if PIPELINE_DIR not in sys.path:
    sys.path.append(PIPELINE_DIR)

from utils import extract_boxed_answers, load_jsonl, strip_think, write_json  # noqa: E402


def _extract_problem_number(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    match = re.search(r"(\d+)\s*$", str(value))
    if not match:
        return None
    return int(match.group(1))


def _parse_judge_score(text):
    if not text:
        return None
    text = strip_think(text)
    scores = [s.strip() for s in extract_boxed_answers(text) if s.strip()]
    if not scores:
        return None
    raw = scores[-1]
    try:
        value = float(raw)
    except ValueError:
        match = re.search(r"-?\d+(?:\.\d+)?", raw)
        if not match:
            return None
        value = float(match.group(0))
    if value in (0.0, 1.0, 6.0, 7.0):
        return int(value)
    return None


def main(args):
    rows = load_jsonl(args.input_file)
    if not rows:
        raise RuntimeError(f"No rows found in {args.input_file}")

    problem_stats = {}
    total_valid = 0
    total_invalid = 0
    total_sum = 0.0

    for row in rows:
        problem_idx = row.get("problem_idx") or row.get("problem_id") or "unknown"
        stats = problem_stats.setdefault(
            problem_idx,
            {
                "problem_number": None,
                "source_name": None,
                "num_rows": 0,
                "valid_scores": 0,
                "invalid_scores": 0,
                "score_sum": 0.0,
                "score_counts": {},
            },
        )
        if stats["problem_number"] is None:
            stats["problem_number"] = row.get("problem_number") or _extract_problem_number(problem_idx)
        if stats["source_name"] is None:
            stats["source_name"] = row.get("source_name")

        stats["num_rows"] += 1
        text = row.get("rating_text") or row.get("generation") or ""
        score = _parse_judge_score(text)
        if score is None:
            stats["invalid_scores"] += 1
            total_invalid += 1
            continue

        stats["valid_scores"] += 1
        total_valid += 1
        stats["score_sum"] += score
        total_sum += score
        key = str(score)
        stats["score_counts"][key] = stats["score_counts"].get(key, 0) + 1

    entries = []
    for problem_idx, stats in problem_stats.items():
        valid = stats["valid_scores"]
        mean_score = stats["score_sum"] / valid if valid else 0.0
        entries.append(
            {
                "problem_idx": problem_idx,
                "problem_number": stats["problem_number"],
                "source_name": stats["source_name"],
                "num_rows": stats["num_rows"],
                "valid_scores": stats["valid_scores"],
                "invalid_scores": stats["invalid_scores"],
                "score_counts": stats["score_counts"],
                "mean_score": mean_score,
            }
        )

    entries.sort(key=lambda item: (item.get("problem_number") is None, item.get("problem_number", 0)))

    category_totals = {"basic": {"sum": 0.0, "count": 0}, "advanced": {"sum": 0.0, "count": 0}}
    for entry in entries:
        problem_idx = entry.get("problem_idx", "")
        if not isinstance(problem_idx, str):
            continue
        key = None
        if "Basic" in problem_idx:
            key = "basic"
        elif "Advanced" in problem_idx:
            key = "advanced"
        if key:
            category_totals[key]["sum"] += entry.get("mean_score", 0.0)
            category_totals[key]["count"] += 1

    problem_means = [entry["mean_score"] for entry in entries]
    metrics = {
        "num_rows": len(rows),
        "num_problems": len(entries),
        "valid_scores": total_valid,
        "invalid_scores": total_invalid,
        "overall": {
            "mean_score": total_sum / total_valid if total_valid else 0.0,
            "mean_score_per_problem": sum(problem_means) / len(problem_means) if problem_means else 0.0,
        },
        "categories": {
            key: {
                "num_problems": value["count"],
                "mean_score": (value["sum"] / value["count"]) if value["count"] else 0.0,
                "mean_score_pct": ((value["sum"] / value["count"]) / 7.0) if value["count"] else 0.0,
            }
            for key, value in category_totals.items()
            if value["count"]
        },
        "problems": entries,
    }

    write_json(args.output_file, metrics)
    print(f"[compute_judge_metrics] Wrote metrics -> {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    main(parser.parse_args())
