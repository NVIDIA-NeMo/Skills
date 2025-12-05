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
import os
from pathlib import Path

from datasets import load_dataset

DEFAULT_SPLITS = [
    ("24q4", "quater_2024_10_12", 207),
    ("25q1", "quater_2025_1_3", 166),
    ("25q2", "quater_2025_4_6", 167),
    ("25q3", "quater_2025_7_9", 144),
]

if __name__ == "__main__":
    if not os.environ.get("HF_TOKEN"):
        raise ValueError("HF_TOKEN environment variable required for LiveCodeBench-Pro download.")

    data_dir = Path(__file__).absolute().parent
    for tag, split, sample_size in DEFAULT_SPLITS:
        dataset = load_dataset("QAQAQAQAQ/LiveCodeBench-Pro", split=split, token=os.environ["HF_TOKEN"])
        assert len(dataset) == sample_size
        output_file = str(data_dir / f"test_{tag}.jsonl")
        with open(output_file, "w") as f:
            for row in dataset:
                output_record = dict()
                output_record["task_id"] = row["problem_id"]
                output_record["question"] = row["problem_statement"]
                output_record["subset_for_metrics"] = row["difficulty"]
                f.write(json.dumps(output_record) + "\n")
