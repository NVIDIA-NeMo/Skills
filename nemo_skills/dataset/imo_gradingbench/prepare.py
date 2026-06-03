#!/usr/bin/env python3
"""
Prepare IMO GradingBench dataset.

Downloads the gradingbench.csv from the DeepMind superhuman repository
and converts it to JSONL format for evaluation.

Source: https://github.com/google-deepmind/superhuman/blob/main/imobench/gradingbench.csv

Task: Given a math problem and a student's solution, predict the grade.
- Model predicts one of 4 categories: 7 (Correct), 6 (Almost), 1 (Partial), 0 (Incorrect)
- Accuracy: Predicted vs Reward category (mapped to 7,6,1,0)
- MAE: |Predicted - actual Points (0-7)|
"""

import csv
import json
import urllib.request
from pathlib import Path

# URL for the raw CSV data
DATA_URL = "https://raw.githubusercontent.com/google-deepmind/superhuman/main/imobench/gradingbench.csv"

# Reward to numeric score mapping
REWARD_TO_SCORE = {
    "Correct": "7",
    "Almost": "6",
    "Partial": "1",
    "Incorrect": "0",
}


def download_csv(url: str, output_path: Path) -> None:
    """Download CSV file from URL."""
    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, output_path)
    print(f"Downloaded to {output_path}")


def process_csv(csv_path: Path, output_path: Path) -> dict:
    """
    Process CSV and convert to JSONL format.

    Returns statistics about the processed data.
    """
    stats = {
        "total": 0,
        "by_reward": {},
        "by_points": {},
    }

    with open(csv_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)

        for row in reader:
            # Extract fields
            grading_id = row["Grading ID"]
            problem_id = row["Problem ID"]
            problem = row["Problem"]
            solution = row["Solution"]
            grading_guidelines = row["Grading guidelines"]
            response = row["Response"]
            points = int(row["Points"])
            reward = row["Reward"]
            problem_source = row["Problem Source"]

            # Map reward to expected answer (for accuracy)
            expected_answer = REWARD_TO_SCORE.get(reward)
            if expected_answer is None:
                print(f"Warning: Unknown reward '{reward}' for {grading_id}, skipping")
                continue

            # Create output record
            record = {
                "problem_id": grading_id,
                "original_problem_id": problem_id,
                "problem": problem,
                "response": response,
                "expected_answer": expected_answer,
                "reward_label": reward,
                "points": points,
                "problem_source": problem_source,
                # Store solution and guidelines for reference (not used in evaluation)
                "reference_solution": solution,
                "grading_guidelines": grading_guidelines,
            }

            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Update stats
            stats["total"] += 1
            stats["by_reward"][reward] = stats["by_reward"].get(reward, 0) + 1
            stats["by_points"][points] = stats["by_points"].get(points, 0) + 1

    return stats


def main():
    # Determine output directory (same as this script)
    script_dir = Path(__file__).parent
    csv_path = script_dir / "gradingbench.csv"
    output_path = script_dir / "test.jsonl"

    # Download CSV if not exists
    if not csv_path.exists():
        download_csv(DATA_URL, csv_path)
    else:
        print(f"CSV already exists: {csv_path}")

    # Process CSV to JSONL
    print("Processing CSV to JSONL...")
    stats = process_csv(csv_path, output_path)

    # Print statistics
    print(f"\n{'=' * 50}")
    print("IMO GradingBench Dataset Prepared")
    print(f"{'=' * 50}")
    print(f"Output: {output_path}")
    print(f"Total examples: {stats['total']}")
    print("\nDistribution by Reward (for Accuracy):")
    for reward, count in sorted(stats["by_reward"].items()):
        score = REWARD_TO_SCORE[reward]
        print(f"  {reward} (→{score}): {count}")
    print("\nDistribution by Points (for MAE):")
    for points, count in sorted(stats["by_points"].items()):
        print(f"  {points}: {count}")

    # Clean up CSV
    # csv_path.unlink()  # Uncomment to delete CSV after processing

    print("\nDone!")


if __name__ == "__main__":
    main()
