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

import json
from pathlib import Path

from datasets import load_dataset

benchmark_file_names = {
    "single_line": "HumanEval-SingleLineInfilling",
    "multi_line": "HumanEval-MultiLineInfilling",
    "random_span": "HumanEval-RandomSpanInfilling",
    "random_span_light": "HumanEval-RandomSpanInfillingLight",
}


def parse_data(split):
    data = load_dataset("loubnabnl/humaneval_infilling", benchmark_file_names[split], split="test")
    return data


def clean_data(dataset, split):
    def map_fn(data):
        data["prefix"] = data.pop("prompt")
        data["language"] = "python"
        data["split"] = split
        data["comment_delimiter"] = "#"
        return data

    dataset = dataset.map(map_fn, remove_columns=["entry_point", "test"])
    return dataset


if __name__ == "__main__":
    data_dir = Path(__file__).absolute().parent
    data_dir.mkdir(exist_ok=True)

    for subset in ["single_line", "multi_line", "random_span"]:
        data = parse_data(split=subset)
        data = clean_data(data, subset)
        print("Len of data: ", len(data))

        output_file_path = data_dir / f"{subset}.jsonl"
        with open(output_file_path, "w") as f:
            for problem in data:
                json.dump(problem, f)
                f.write("\n")
