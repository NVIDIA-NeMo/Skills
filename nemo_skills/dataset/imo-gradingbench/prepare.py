# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import csv
import io
import json
import urllib.request
from pathlib import Path

if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)

    base_url = "https://raw.githubusercontent.com/google-deepmind/superhuman/c1ee02e03d4cdb2ab21cd01ac927d895f5287fc8/imobench"
    source_url = f"{base_url}/gradingbench.csv"
    output_file = data_dir / "test.jsonl"

    with urllib.request.urlopen(source_url, timeout=30) as response:
        content = response.read().decode("utf-8")
        reader = csv.DictReader(io.StringIO(content))

        with open(output_file, "w", encoding="utf-8") as out:
            for row in reader:
                entry = {
                    "grading_id": row["Grading ID"],
                    "problem_id": row["Problem ID"],
                    "problem_statement": row["Problem"],
                    "reference_solution": row["Solution"],
                    "rubric": row["Grading guidelines"],
                    "proof": row["Response"],
                    "points": row["Points"],
                    "reward": row["Reward"],
                    "source": row["Problem Source"],
                    # We set 'expected_answer' to the reward/points for evaluation
                    "expected_answer": row["Reward"],
                }
                out.write(json.dumps(entry) + "\n")
