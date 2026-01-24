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

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

# Github paths for BFCL
# using same setup as in the nemo-skills container for consistency
REPO_URL = "https://github.com/ShishirPatil/gorilla.git"
BFCL_GIT_COMMIT = "86d0374d0db52623c5092a73f82c22b87b7e9a25"
BFCL_EVAL_SUBDIR = "berkeley-function-call-leaderboard"
BFCL_EXTRA_INDEX_URL = "https://download.pytorch.org/whl/cpu"


# TODO: we should probably move all such runtime installations to an isolated venv..
def ensure_bfcl_eval_installed():
    try:
        import bfcl_eval  # noqa: F401

        return
    except (ModuleNotFoundError, ImportError, AttributeError):
        logging.getLogger(__name__).info("bfcl_eval not found. Installing runtime dependency...")
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = Path(temp_dir) / "gorilla"
            subprocess.run(["git", "clone", REPO_URL, str(repo_dir)], check=True)
            subprocess.run(["git", "checkout", BFCL_GIT_COMMIT], check=True, cwd=str(repo_dir))
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-cache-dir",
                    "-e",
                    str(repo_dir / BFCL_EVAL_SUBDIR),
                    "--extra-index-url",
                    BFCL_EXTRA_INDEX_URL,
                ],
                check=True,
            )


ensure_bfcl_eval_installed()

from bfcl_eval.constants.category_mapping import MEMORY_SCENARIO_NAME  # noqa: E402
from bfcl_eval.utils import (  # noqa: E402
    is_agentic,
    is_format_sensitivity,
    is_memory,
    is_multi_turn,
    is_web_search,
    load_file,
    populate_initial_settings_for_memory_test_cases,
    populate_initial_settings_for_web_search_test_cases,
    populate_test_cases_with_predefined_functions,
    process_agentic_test_case,
    process_memory_test_case,
    process_web_search_test_case,
)

from nemo_skills.dataset.bfcl_v3.constants import (  # noqa: E402
    ALL_SCORING_CATEGORIES,
    DATA_FOLDER_PATH,
    VERSION_PREFIX,
)
from nemo_skills.dataset.bfcl_v3.utils import (  # noqa: E402
    convert_to_tool,
    func_doc_language_specific_pre_processing,
)
from nemo_skills.utils import get_logger_name  # noqa: E402

LOG = logging.getLogger(get_logger_name(__file__))

# Define the configuration as a dictionary
DEFAULT_SETTINGS = """
DATASET_GROUP = "tool"
METRICS_TYPE = "bfcl"
GENERATION_ARGS = "++eval_type=bfcl"
GENERATION_MODULE = "nemo_skills.inference.eval.bfcl"
"""


# Adapted from - https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl_eval/utils.py#L403
def process_multi_turn_test_case(instance):
    """
    Multi-turn test cases don't have the function doc in the prompt. We need to add them here.
    """
    # Mark whether the instance is single-turn or multi-turn.
    # This is used to determine if the inference should be done in a single turn or multiple turns.
    if not is_multi_turn(instance["id"]) and not is_agentic(instance["id"]):
        instance["single_turn"] = True
    else:
        instance["single_turn"] = False

    return instance


def load_dataset_entry(
    target_folder: Path,
    test_category: str,
    include_prereq: bool = True,
) -> list[dict]:
    """
    This function retrieves the dataset entry for a given test category.
    The input should not be a test category goup, but a specific test category.
    If `include_prereq` is True, it will include the pre-requisite entries for the memory test categories.
    """
    # Skip for now
    if is_format_sensitivity(test_category):
        return []
        # Format sensitivity categories
        # all_entries = load_format_sensitivity_test_cases()

    elif is_web_search(test_category):
        # Web search categories
        file_name = f"{VERSION_PREFIX}_web_search.json"
        all_entries = load_file(target_folder / file_name)
        all_entries = process_web_search_test_case(all_entries, test_category)

    elif is_memory(test_category):
        # Memory categories
        all_entries = load_file(target_folder / f"{VERSION_PREFIX}_memory.json")
        for scenario in MEMORY_SCENARIO_NAME:
            all_entries = process_memory_test_case(all_entries, test_category, scenario, include_prereq=include_prereq)
    else:
        # All other categories, we don't need any special handling
        file_name = f"{VERSION_PREFIX}_{test_category}.json"
        all_entries = load_file(target_folder / file_name)

    all_entries = process_agentic_test_case(all_entries)
    all_entries = populate_test_cases_with_predefined_functions(all_entries)
    all_entries = [process_multi_turn_test_case(entry) for entry in all_entries]

    all_entries = populate_initial_settings_for_memory_test_cases(all_entries, str(target_folder))
    all_entries = populate_initial_settings_for_web_search_test_cases(all_entries)

    # Convert function calls to tools format and add them to the system prompt
    for instance in all_entries:
        if "function" in instance:
            # Add the tools to the system prompt
            instance["function"] = func_doc_language_specific_pre_processing(instance["function"], test_category)
            instance["tools"] = convert_to_tool(instance["function"])

    return all_entries


# NOTE: This function was unified to handle both BFCLv3 and BFCLv4 data
def download_and_process_bfcl_data(repo_url, subfolder_path, output_dir, scoring_categories):
    """
    Download JSON files from the BFCL GitHub repo via cloning

    Args:
        repo_url: GitHub repository URL
        subfolder_path: Path to the data subfolder in case of BFCL
        output_dir: Directory to save the processed JSONL files
        file_prefix: Only process files starting with this prefix
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Clone repository with minimal depth
            LOG.info(f"Cloning repository {repo_url} to {temp_dir}")
            subprocess.run(["git", "clone", "--depth=1", repo_url, temp_dir], check=True, capture_output=True)

            # Find the target folder
            target_folder = Path(temp_dir) / subfolder_path

            if not os.path.exists(target_folder):
                LOG.error(f"Folder {subfolder_path} not found in repository")
                raise FileNotFoundError(
                    f"Folder {subfolder_path} not found in {repo_url} cloned to {temp_dir}. The structure of BFCL has changed!"
                )

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            processed_categories = 0
            for test_category in scoring_categories:
                split_dirname = os.path.join(output_dir, test_category)
                if not os.path.exists(split_dirname):
                    os.makedirs(split_dirname)

                with open(os.path.join(split_dirname, "__init__.py"), "w") as f:
                    f.write(DEFAULT_SETTINGS)

                output_file = os.path.join(split_dirname, "test.jsonl")
                test_entries = load_dataset_entry(target_folder, test_category)
                with open(output_file, "w") as f_out:
                    for instance in test_entries:
                        f_out.write(json.dumps(instance) + "\n")

                processed_categories += 1

            LOG.info(f"Successfully processed {processed_categories} BFCLv4 categories to {output_dir}")

        except subprocess.CalledProcessError:
            LOG.exception("Git command failed")
            LOG.error("Make sure git is installed and the repository URL is correct")


def main(args):
    LOG.warning(
        "Currently processing according to the OpenAI model style which works for most models, including Qwen/Nemotron/DeepSeek."
    )

    download_and_process_bfcl_data(
        REPO_URL,
        DATA_FOLDER_PATH,
        output_dir=os.path.join(os.path.dirname(__file__)),
        scoring_categories=ALL_SCORING_CATEGORIES,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default=None, required=False)
    args = parser.parse_args()

    main(args)
