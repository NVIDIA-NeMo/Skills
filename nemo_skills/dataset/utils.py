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

import contextlib
import importlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict
from urllib.error import URLError

from nemo_skills.evaluation.math_grader import extract_answer


@contextlib.contextmanager
def add_to_path(p):
    old_path = sys.path
    sys.path = sys.path[:]
    sys.path.insert(0, str(p))
    try:
        yield
    finally:
        sys.path = old_path


def add_rounding_instruction(data: Dict) -> Dict:
    try:
        float(data['expected_answer'])
        number_of_values = 0
        if '.' in str(data['expected_answer']):
            number_of_values = len(str(data['expected_answer']).split('.')[1])
        if number_of_values == 0:
            data['problem'] += ' Express the answer as an integer.'
        elif number_of_values == 1:
            data['problem'] += ' Round the answer to one decimal place.'
        else:
            data['problem'] += f' Round the answer to {number_of_values} decimal places.'
    except ValueError:
        pass
    return data


def get_dataset_module(dataset, extra_datasets=None):
    try:
        dataset_module = importlib.import_module(f"nemo_skills.dataset.{dataset}")
        found_in_extra = False
    except ModuleNotFoundError:
        extra_datasets = extra_datasets or os.environ.get("NEMO_SKILLS_EXTRA_DATASETS")
        if extra_datasets is None:
            raise
        with add_to_path(extra_datasets):
            dataset_module = importlib.import_module(dataset)
        found_in_extra = True
    return dataset_module, found_in_extra


def get_lean4_header():
    LEAN4_HEADER = "import Mathlib\n\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen Topology Filter Real Complex TopologicalSpace Finset Function Metric Nat Rat\nopen scoped BigOperators Matrix\n\n"
    return LEAN4_HEADER


def add_header_to_jsonl_inplace(jsonl_path, header):
    """
    Adds or updates the header field for all entries in a JSONL file.

    Args:
        jsonl_path (str): Path to the JSONL file.
        header (str): The header string to add or update in each entry.
    """
    with open(jsonl_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    with open(jsonl_path, "w", encoding="utf-8") as file:
        for line in lines:
            data = json.loads(line)
            data["header"] = header
            file.write(json.dumps(data) + "\n")


def download_with_retries(url, output_file, max_retries=3, retry_delay=1):
    """Download a file with retry logic."""
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, output_file)
            return True
        except URLError as e:
            if attempt == max_retries - 1:
                raise RuntimeError(f"Failed to download after {max_retries} attempts: {e}")
            time.sleep(retry_delay * (attempt + 1))
    return False


def save_data_from_qwen(dataset, split="test"):
    url = (
        "https://raw.githubusercontent.com/QwenLM/Qwen2.5-Math/refs/heads/main/evaluation/data/{dataset}/{split}.jsonl"
    )

    data_dir = Path(__file__).absolute().parent
    original_file = str(data_dir / dataset / f"original_{split}.json")
    data_dir.mkdir(exist_ok=True)
    output_file = str(data_dir / dataset / f"{split}.jsonl")
    data = []
    if not os.path.exists(original_file):
        formatted_url = url.format(split=split, dataset=dataset)
        download_with_retries(formatted_url, original_file)

    with open(original_file, "rt", encoding="utf-8") as fin:
        for line in fin:
            entry = json.loads(line)

            if "answer" in entry:
                entry["expected_answer"] = entry.pop("answer")

            if "problem" not in entry:
                entry["problem"] = entry.pop("question")

            if dataset == "olympiadbench":
                entry["expected_answer"] = entry.pop("final_answer")[0].strip("$")

            if dataset == "minerva_math":
                entry["expected_answer"] = extract_answer(entry["solution"])

            data.append(entry)

    with open(output_file, "wt", encoding="utf-8") as fout:
        for entry in data:
            fout.write(json.dumps(entry) + "\n")

    # cleaning up original data file
    os.remove(original_file)

    return output_file


def get_mcq_fields(question, choices):
    options_dict = {chr(ord('A') + i): option for i, option in enumerate(choices)}
    options_text = "\n".join(f"{letter}. {option}" for letter, option in options_dict.items())
    return {
        "problem": f"{question}\n{options_text}",
        "options": options_text,
        **options_dict,
    }
