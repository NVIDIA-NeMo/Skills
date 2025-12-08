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
from tqdm import tqdm


def write_data_to_file(output_file, data):
    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in tqdm(data, desc=f"Writing {output_file.name}"):
            entry["expected_answer"] = entry.pop("answer")
            json.dump(entry, fout)
            fout.write("\n")


if __name__ == "__main__":
    dataset = load_dataset("MathArena/apex-shortlist")["train"]
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / "test.jsonl"
    write_data_to_file(output_file, dataset)
