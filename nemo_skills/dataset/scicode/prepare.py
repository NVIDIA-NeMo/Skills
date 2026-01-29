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
from pathlib import Path

from datasets import load_dataset

if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)

    dataset = load_dataset("SciCode1/SciCode")

    split_mapping = {"validation": "dev", "test": "test"}
    test_aai_file = data_dir / "test_aai.jsonl"
    with open(test_aai_file, "w", encoding="utf-8") as test_aai_fout:
        for hf_split, output_split in split_mapping.items():
            output_file = data_dir / f"{output_split}.jsonl"
            with open(output_file, "wt", encoding="utf-8") as fout:
                for entry in dataset[hf_split]:
                    line = json.dumps(entry) + "\n"
                    fout.write(line)
                    test_aai_fout.write(line)
