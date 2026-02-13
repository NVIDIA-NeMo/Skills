import json
from pathlib import Path

if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    with open(data_dir / "test.jsonl", "wt", encoding="utf-8") as fout:
        fout.write(
            json.dumps(
                {
                    "problem": "What is 2 + 2?",
                    "expected_answer": 4,
                }
            )
            + "\n"
        )
