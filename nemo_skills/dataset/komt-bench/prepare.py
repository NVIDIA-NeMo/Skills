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
    "https://raw.githubusercontent.com/LG-AI-EXAONE/KoMT-Bench/main/"
    "FastChat/fastchat/llm_judge/data/mt_bench/question_ko.jsonl"
)
URL_REF_ANSWERS = (
    "https://raw.githubusercontent.com/LG-AI-EXAONE/KoMT-Bench/main/"
    "FastChat/fastchat/llm_judge/data/mt_bench/reference_answer_ko/gpt-4.jsonl"
)

# qids 138/140 expect JSON/CSV output; exempted by upstream detector.py.
LANG_PENALTY_EXEMPT_QIDS = {138, 140}


if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)

    questions_file = data_dir / "question_ko.jsonl"
    ref_file = data_dir / "gpt-4-ref-ko.jsonl"
    output_file = data_dir / "test.jsonl"

    urllib.request.urlretrieve(URL_QUESTIONS, str(questions_file))
    urllib.request.urlretrieve(URL_REF_ANSWERS, str(ref_file))

    ref_by_qid = {}
    with open(ref_file, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            ref_by_qid[data["question_id"]] = data["choices"][0]["turns"]

    with open(questions_file, "rt", encoding="utf-8") as fin, open(output_file, "wt", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            q = json.loads(line)
            qid = q["question_id"]
            turns = q["turns"]
            if len(turns) != 2:
                raise ValueError(f"KoMT-Bench question {qid} expected 2 turns, got {len(turns)}")
            ref_turns = ref_by_qid.get(qid, [None, None])
            entry = {
                "question_id": qid,
                "category": q["category"],
                "question_1": turns[0],
                "question_2": turns[1],
                "ref_answer_1": ref_turns[0],
                "ref_answer_2": ref_turns[1],
            }
            if qid not in LANG_PENALTY_EXEMPT_QIDS:
                entry["target_language"] = "ko"
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
