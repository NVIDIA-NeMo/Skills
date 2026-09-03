# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Materialize the SWE-bench Lite split as a local JSONL file.

Downloads the benchmark from the Hugging Face Hub and writes one JSON record per
instance to `test.jsonl`, the format the SHERLOC generation and evaluation steps
read. Record fields are kept verbatim, so the gold patch used to derive reference
edit locations stays available downstream.
"""

import argparse
import json

from datasets import load_dataset

parser = argparse.ArgumentParser(description="Convert a Hugging Face dataset to a JSONL file.")
parser.add_argument("--dataset_name", type=str, default="princeton-nlp/SWE-bench_Lite")
parser.add_argument("--split", type=str, default="test")
parser.add_argument("--output_file", type=str, default="test.jsonl")
args = parser.parse_args()

dataset = load_dataset(args.dataset_name, split=args.split)

with open(args.output_file, "w", encoding="utf-8") as f:
    for record in dataset:
        f.write(json.dumps(record) + "\n")

print(f"Dataset '{args.dataset_name}' (split: {args.split}) converted to '{args.output_file}'.")
