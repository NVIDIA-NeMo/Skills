import json
import argparse

from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset

TOPIC_TO_SPLIT_MAP = {
    "Humanities and Social Sciences": "humanities",
    "Health": "health",
    "Software Engineering": "swe",
    "Science Engineering and Mathematics": "stem",
    "Law": "law",
    "Finance": "finance",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--splits', default=['all', 'humanities', 'health', 'swe', 'stem', 'law', 'finance'], nargs="+", choices=["all", "humanities", "health", "swe", "stem", "law", "finance"])
    return parser.parse_args()

def format_entry(entry) -> dict:
    return {
        'id': entry['question_id'],
        'domain': entry['domain'],
        'topic': entry['topic'],
        'question': entry['question'],
        'target': entry['answer']
    }

def write_jsonl(data: list[dict], path: str):
    with open(path, 'w', encoding='utf-8') as f:
        for d in data:
            f.write(json.dumps(d) + "\n")

if __name__ == "__main__":
    args = parse_args()

    dataset = load_dataset("ArtificialAnalysis/AA-Omniscience-Public", split="train")
    jsonl_data = [format_entry(d) for d in dataset]
    output_dir = Path(__file__).absolute().parent

    split_set = set(args.splits)
    splits = {'all': dataset, **{TOPIC_TO_SPLIT_MAP.get(t, str(t).lower()): dataset.filter(lambda x: x['domain'] == t) for t in dataset.unique('domain')}}
    splits = {k: v for k, v in splits.items() if k in split_set}

    for split, data in splits.items():
        output_file = output_dir / f"{split}.jsonl"
        formatted_data = [format_entry(entry) for entry in data]
        write_jsonl(formatted_data, output_file)
