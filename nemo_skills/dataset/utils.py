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
import json
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Dict
from urllib.error import URLError

from nemo_skills.evaluation.math_grader import extract_answer


def add_rounding_instruction(data: Dict) -> Dict:
    try:
        float(data["expected_answer"])
        number_of_values = 0
        if "." in str(data["expected_answer"]):
            number_of_values = len(str(data["expected_answer"]).split(".")[1])
        if number_of_values == 0:
            data["problem"] += " Express the answer as an integer."
        elif number_of_values == 1:
            data["problem"] += " Round the answer to one decimal place."
        else:
            data["problem"] += f" Round the answer to {number_of_values} decimal places."
    except ValueError:
        pass
    return data


def import_from_path(file_path, module_name=None):
    if module_name is None:  # unique random name
        module_name = f"dynamic_module_{int(time.time() * 1000)}"
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def get_dataset_name(dataset):
    """Extract the canonical dataset name from a dataset identifier (short name or path)."""
    if "/" in dataset:
        return Path(dataset).name
    return dataset


def get_dataset_path(dataset):
    """Resolve a dataset identifier to its directory path.

    Resolution order:
    1. If dataset contains '/', treat as a direct path.
    2. Check extra benchmark map.
    3. Fall back to built-in dataset directory.
    """
    if "/" in dataset:
        return Path(dataset)
    extra_map = get_extra_benchmark_map()
    if dataset in extra_map:
        return Path(extra_map[dataset])
    return Path(__file__).parent / dataset


def get_extra_benchmark_map():
    """Load extra benchmark map from NEMO_SKILLS_EXTRA_BENCHMARK_MAP env var if set.

    Returns a dict mapping short names to directory paths, or empty dict.
    Relative paths in the map are resolved relative to the map file's directory.
    """
    map_path = os.environ.get("NEMO_SKILLS_EXTRA_BENCHMARK_MAP")
    if not map_path:
        return {}
    map_dir = Path(map_path).resolve().parent
    with open(map_path, "r") as f:
        raw_map = json.load(f)
    return {
        name: str((map_dir / path).resolve()) if not os.path.isabs(path) else path for name, path in raw_map.items()
    }


def _load_external_dataset(dataset_path):
    """Load dataset module from an external directory containing __init__.py."""
    dataset_path = Path(dataset_path)
    init_path = dataset_path / "__init__.py"
    if not init_path.exists():
        raise RuntimeError(f"Expected {init_path} to exist for external dataset {dataset_path}")
    dataset_module = import_from_path(str(init_path))
    # parent of benchmark dir so that data_path/benchmark_name/split.jsonl works
    data_path = str(dataset_path.parent)
    return dataset_module, data_path


def get_default_dataset_module(dataset):
    data_path = "/nemo_run/code/nemo_skills/dataset"
    dataset_module = importlib.import_module(f"nemo_skills.dataset.{dataset}")

    return dataset_module, data_path


def get_dataset_module(dataset):
    """Get dataset module from nemo_skills.dataset, extra benchmark map, or a directory path.

    Resolution order:
    1. If dataset contains '/', treat as a direct path to a directory with __init__.py.
    2. Otherwise, check both built-in datasets and the extra benchmark map.
       - If found in both, raise an error (ambiguous).
       - If found in exactly one, use it.
       - If found in neither, raise an error.
    """
    if "/" in dataset:
        return _load_external_dataset(dataset)

    # Check built-in
    found_builtin = True
    try:
        dataset_module, data_path = get_default_dataset_module(dataset)
    except ModuleNotFoundError:
        found_builtin = False

    # Check extra benchmark map
    extra_map = get_extra_benchmark_map()
    found_in_map = dataset in extra_map

    if found_builtin and found_in_map:
        raise RuntimeError(
            f"Dataset '{dataset}' found both as a built-in dataset and in the extra benchmark map "
            f"(pointing to {extra_map[dataset]}). Please use the full path to resolve the ambiguity."
        )

    if found_in_map:
        return _load_external_dataset(extra_map[dataset])

    if found_builtin:
        return dataset_module, data_path

    map_path = os.environ.get("NEMO_SKILLS_EXTRA_BENCHMARK_MAP")
    if map_path:
        raise RuntimeError(f"Dataset '{dataset}' not found in built-in datasets or extra benchmark map at {map_path}.")
    raise RuntimeError(
        f"Dataset '{dataset}' not found in built-in datasets. Did you forget to set NEMO_SKILLS_EXTRA_BENCHMARK_MAP?"
    )


def get_lean4_header():
    LEAN4_HEADER = "import Mathlib\n\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen Topology Filter Real Complex TopologicalSpace Finset Function Metric Nat Rat\nopen scoped BigOperators Matrix\n\n"
    return LEAN4_HEADER


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
    ns_dataset = dataset if dataset != "math" else "hendrycks_math"
    original_file = str(data_dir / ns_dataset / f"original_{split}.json")
    data_dir.mkdir(exist_ok=True)
    output_file = str(data_dir / ns_dataset / f"{split}.jsonl")
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
    options_dict = {chr(ord("A") + i): option for i, option in enumerate(choices)}
    options_text = "\n".join(f"{letter}) {option}" for letter, option in options_dict.items())
    question = question.strip("\n")
    return {
        "problem": f"{question}\n\n{options_text}",
        "options": options_text,
        **options_dict,
    }
