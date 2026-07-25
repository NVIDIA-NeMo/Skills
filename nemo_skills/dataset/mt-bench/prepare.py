# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

URL_QUESTIONS = (
    "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/question.jsonl"
)
URL_REF_ANSWERS = "https://raw.githubusercontent.com/lm-sys/FastChat/main/fastchat/llm_judge/data/mt_bench/reference_answer/gpt-4.jsonl"


if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)

    questions_file = data_dir / "question.jsonl"
    ref_file = data_dir / "reference_answer.jsonl"
    output_file = data_dir / "test.jsonl"

    urllib.request.urlretrieve(URL_QUESTIONS, str(questions_file))
    urllib.request.urlretrieve(URL_REF_ANSWERS, str(ref_file))

    # Build reference answer lookup {question_id: {turn1: str, turn2: str}}
    ref_answers = {}
    with open(ref_file, "rt", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            qid = data["question_id"]
            turns = data["choices"][0]["turns"]
            ref_answers[qid] = {"turn1": turns[0], "turn2": turns[1] if len(turns) > 1 else None}

    with open(questions_file, "rt", encoding="utf-8") as fin, open(output_file, "wt", encoding="utf-8") as fout:
        for line in fin:
            data = json.loads(line)
            qid = data["question_id"]
            turns = data["turns"]
            if len(turns) != 2:
                raise ValueError(f"MT-Bench question {qid} expected 2 turns, got {len(turns)}")
            # Reference answers are populated upstream only for math/reasoning/coding;
            # other categories legitimately have none.
            ref = ref_answers.get(qid, {})
            entry = {
                "question_id": qid,
                "category": data["category"],
                "question_1": turns[0],
                "question_2": turns[1],
                "ref_answer_1": ref.get("turn1"),
                "ref_answer_2": ref.get("turn2"),
            }
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
