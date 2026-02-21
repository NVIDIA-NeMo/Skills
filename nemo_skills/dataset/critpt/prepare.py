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
from pathlib import Path

from datasets import load_dataset

if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    output_file = str(data_dir / "test.jsonl")

    # Load the CritPt dataset from Hugging Face
    dataset = load_dataset("CritPt-Benchmark/CritPt", split="train")

    with open(output_file, "wt", encoding="utf-8") as fout:
        for problem in dataset:
            entry = {
                "problem_id": problem["problem_id"],
                "problem": problem["problem_description"],
                "code_template": problem["code_template"],
                "expected_answer": "",
                "metadata": {
                    "notebook_path": problem["metadata_notebook_path"],
                    "problem_setup": problem["metadata_problem_setup"],
                },
            }
            fout.write(json.dumps(entry) + "\n")
