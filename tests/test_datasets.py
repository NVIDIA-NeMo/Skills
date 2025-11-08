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

import importlib

# tuple of dataset name, available splits and prepared sft files
DATASETS = [
    ("aime25", ["test"]),
    ("math-500", ["test"]),
    ("aime24", ["test"]),
    ("amc23", ["test"]),
    ("omni-math", ["test"]),
    ("algebra222", ["test"]),
    ("arena-hard", ["test"]),
    ("asdiv", ["test"]),
    ("gsm-plus", ["test", "test_rounded"]),
    ("gsm8k", ["train", "test"]),
    ("hle", ["math", "text"]),
    ("simpleqa", ["test", "verified"]),
    ("human-eval", ["test"]),
    (
        "livecodebench",
        [
            "test_v5_2408_2502",
            "test_v5_2410_2502",
            "test_v5_2410_2504",
            "test_v6_2408_2502",
            "test_v6_2410_2502",
            "test_v6_2410_2504",
        ],
    ),
    ("ifeval", ["test"]),
    ("hendrycks_math", ["train", "test"]),
    ("math-odyssey", ["test"]),
    ("mawps", ["test"]),
    ("mbpp", ["test"]),
    ("mmlu", ["test", "dev", "val"]),
    ("svamp", ["test"]),
    ("answer-judge", ["test"]),
    ("mmlu-pro", ["test"]),
    ("mmlu-redux", ["test"]),
    ("gpqa", ["diamond", "main", "extended"]),
    ("minerva_math", ["test"]),
    ("olympiadbench", ["test"]),
    ("gaokao2023en", ["test"]),
    ("college_math", ["test"]),
    ("comp-math-24-25", ["test"]),
    ("mmau-pro", ["test"]),
]


def test_dataset_init_defaults():
    for dataset, _ in DATASETS:
        dataset_module = importlib.import_module(f"nemo_skills.dataset.{dataset}")
        assert hasattr(dataset_module, "DATASET_GROUP"), f"{dataset} is missing DATASET_GROUP attribute"
        assert dataset_module.DATASET_GROUP in [
            "math",
            "code",
            "chat",
            "multichoice",
            "long-context",
            "tool",
            "speechlm",
        ], f"{dataset} has invalid DATASET_GROUP"
