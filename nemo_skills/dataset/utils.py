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
import tempfile
import time
import urllib.request
from enum import Enum
from pathlib import Path
from typing import Dict
from urllib.error import URLError

from nemo_skills.evaluation.math_grader import extract_answer
from nemo_skills.pipeline.utils import cluster_download_file, get_unmounted_path


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


class ExtraDatasetType(str, Enum):
    local = "local"
    cluster = "cluster"


def _get_dataset_module_from_cluster(cluster_config, mounted_path):
    # getting tmp path to download init.py
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = str(Path(tmpdir) / "init.py")
        cluster_dataset_path = get_unmounted_path(cluster_config, mounted_path)
        try:
            cluster_download_file(cluster_config, cluster_dataset_path, tmp_path)
        except FileNotFoundError:
            raise RuntimeError(
                f"Init file {mounted_path} not found on the cluster. "
                f"Please check the dataset name you're using. Did you forget to run prepare data commands?"
            )
        return import_from_path(tmp_path)


def get_default_dataset_module(dataset, data_dir=None, cluster_config=None):
    is_on_cluster = False
    if data_dir is None:
        data_path = "/nemo_run/code/nemo_skills/dataset"
        dataset_module = importlib.import_module(f"nemo_skills.dataset.{dataset}")
    else:
        data_path = data_dir
        if cluster_config is None or cluster_config["executor"] == "none":
            with add_to_path(data_dir):
                dataset_module = importlib.import_module(dataset)
        else:
            if cluster_config["executor"] == "local":
                with add_to_path(get_unmounted_path(cluster_config, data_dir)):
                    dataset_module = importlib.import_module(dataset)
            else:
                dataset = dataset.replace(".", "/")
                dataset_module = _get_dataset_module_from_cluster(cluster_config, f"{data_dir}/{dataset}/__init__.py")
                is_on_cluster = True
    return dataset_module, data_path, is_on_cluster


def get_dataset_module(dataset, data_dir=None, cluster_config=None, extra_datasets=None, extra_datasets_type=None):
    """
    Get dataset module either in default folder or in extra datasets folder.

    If cluster_config is provided, the data_dir will be resolved as a mounted
    path and appropriately downloaded from the cluster.

    Same will be done for extra_datasets if extra_datasets_type is cluster.

    Search priority:
    1. data_dir (or `nemo_skills.dataset` if None) folder
    3. extra_datasets parameter if defined
    4. `NEMO_SKILLS_EXTRA_DATASETS` environment variable
    """
    try:
        dataset_module, data_path, is_on_cluster = get_default_dataset_module(dataset, data_dir, cluster_config)
    except ModuleNotFoundError:
        try:
            dataset = dataset.replace(".", "/")
            extra_datasets = extra_datasets or os.environ.get("NEMO_SKILLS_EXTRA_DATASETS")
            is_on_cluster = False
            data_path = extra_datasets
            if extra_datasets is None:
                raise RuntimeError(f"Dataset {dataset} not found in {data_dir if data_dir else 'nemo_skills.dataset'}")
            if extra_datasets_type == ExtraDatasetType.local:
                with add_to_path(extra_datasets):
                    dataset_module = importlib.import_module(dataset)
            else:
                dataset_module = _get_dataset_module_from_cluster(
                    cluster_config, f'{extra_datasets}/{dataset}/"__init__.py"'
                )
                is_on_cluster = True
        except ModuleNotFoundError:
            raise RuntimeError(
                f"Dataset {dataset} not found in any of the searched locations: "
                f"{data_dir if data_dir else 'nemo_skills.dataset'}, {extra_datasets}"
            )
    return dataset_module, data_path, is_on_cluster


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
