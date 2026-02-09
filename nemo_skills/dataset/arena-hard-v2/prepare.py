# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import urllib.request
from pathlib import Path

URL_QUESTIONS = "https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/data/arena-hard-v2.0/question.jsonl"
# Category-specific baselines as per official arena-hard-auto implementation
URL_BASELINE_HARD_PROMPT = "https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/data/arena-hard-v2.0/model_answer/o3-mini-2025-01-31.jsonl"
URL_BASELINE_CREATIVE_WRITING = "https://raw.githubusercontent.com/lmarena/arena-hard-auto/main/data/arena-hard-v2.0/model_answer/gemini-2.0-flash-001.jsonl"

# Mapping of category to baseline URL
CATEGORY_BASELINES = {
    "hard_prompt": URL_BASELINE_HARD_PROMPT,
    "creative_writing": URL_BASELINE_CREATIVE_WRITING,
}


def extract_answer_text(data):
    """Extract the answer text from the baseline model's response format."""
    messages = data["messages"]
    for msg in messages:
        if msg["role"] == "assistant":
            content = msg["content"]
            return content["answer"] if isinstance(content, dict) else content
    raise ValueError("No assistant message found in the data.")


if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    questions_file = str(data_dir / "question.jsonl")
    output_file = str(data_dir / "test.jsonl")

    # Download questions
    urllib.request.urlretrieve(URL_QUESTIONS, questions_file)

    # Download and process all baseline files
    baseline_answers = {}
    for category, url in CATEGORY_BASELINES.items():
        baseline_file = str(data_dir / f"baseline_{category}.jsonl")
        urllib.request.urlretrieve(url, baseline_file)

        with open(baseline_file, "rt", encoding="utf-8") as fin:
            for line in fin:
                data = json.loads(line)
                uid = data["uid"]
                if uid not in baseline_answers:
                    baseline_answers[uid] = {}
                baseline_answers[uid][category] = extract_answer_text(data)

    # Create test.jsonl with category-specific baseline answers
    with open(questions_file, "rt", encoding="utf-8") as fin, open(output_file, "wt", encoding="utf-8") as fout:
        for line in fin:
            data = json.loads(line)
            data["question"] = data.pop("prompt")
            category = data["category"]
            data["baseline_answer"] = baseline_answers[data["uid"]][category]
            fout.write(json.dumps(data) + "\n")
