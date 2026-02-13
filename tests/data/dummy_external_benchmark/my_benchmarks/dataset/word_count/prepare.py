import json
from pathlib import Path

SAMPLES = [
    {"sentence": "The quick brown fox", "expected_answer": 4},
    {"sentence": "Hello world", "expected_answer": 2},
    {"sentence": "NeMo Skills is great for evaluation", "expected_answer": 6},
    {"sentence": "One", "expected_answer": 1},
    {"sentence": "A B C D E F G", "expected_answer": 7},
]

if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    output_file = data_dir / "test.jsonl"
    with open(output_file, "wt", encoding="utf-8") as fout:
        for sample in SAMPLES:
            fout.write(json.dumps(sample) + "\n")
