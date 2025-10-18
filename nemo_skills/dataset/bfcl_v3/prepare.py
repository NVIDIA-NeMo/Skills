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
import glob
import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from nemo_skills.dataset.bfcl_v3.constants import (
    DATA_FOLDER_PATH,
    MULTI_TURN_FUNC_DOC_FILE_MAPPING,
    MULTI_TURN_FUNC_DOC_PATH,
)
from nemo_skills.dataset.bfcl_v3.utils import (
    convert_to_tool,
    func_doc_language_specific_pre_processing,
    is_multi_turn,
    load_file,
)
from nemo_skills.utils import get_logger_name

LOG = logging.getLogger(get_logger_name(__file__))


# Github paths for BFCL
REPO_URL = "https://github.com/ShishirPatil/gorilla.git"

# Define the configuration as a dictionary
DEFAULT_SETTINGS = """
DATASET_GROUP = "tool"
METRICS_TYPE = "bfcl"
GENERATION_ARGS = "++eval_type=bfcl"
GENERATION_MODULE = "nemo_skills.inference.eval.bfcl"
"""


# Adapted from - https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl_eval/_llm_response_generation.py#L142
def process_multi_turn_test_case(instance, repo_root_dir):
    """
    Multi-turn test cases don't have the function doc in the prompt. We need to add them here.
    """
    # Mark whether the instance is single-turn or multi-turn.
    # This is used to determine if the inference should be done in a single turn or multiple turns.
    if not is_multi_turn(instance["id"]):
        instance["single_turn"] = True
        return instance
    else:
        instance["single_turn"] = False

    involved_classes = instance["involved_classes"]
    instance["function"] = []
    for func_collection in involved_classes:
        # func_doc is a list of dict
        func_doc = load_file(
            repo_root_dir / MULTI_TURN_FUNC_DOC_PATH / MULTI_TURN_FUNC_DOC_FILE_MAPPING[func_collection]
        )
        instance["function"].extend(func_doc)

    # Handle Miss Func category; we need to remove the holdout function doc
    if "missed_function" in instance:
        for turn_index, missed_func_names in instance["missed_function"].items():
            instance["missed_function"][turn_index] = []
            for missed_func_name in missed_func_names:
                for i, func_doc in enumerate(instance["function"]):
                    if func_doc["name"] == missed_func_name:
                        # Add the missed function doc to the missed_function list
                        instance["missed_function"][turn_index].append(func_doc)
                        # Remove it from the function list
                        instance["function"].pop(i)
                        break

    return instance


def process_file(repo_root_dir, input_file, output_file, model_type="llama-nemotron"):
    """Preprocess the functions and convert them to tool format.
    Also mark whether the instance is single-turn or multi-turn which is used during inference.
    """

    with open(input_file, "r") as f, open(output_file, "w") as f_out:
        for idx, line in enumerate(f):
            instance = json.loads(line)
            test_category = instance["id"].rsplit("_", 1)[0]
            if idx == 0:
                LOG.info(f"Processing {test_category}")

            # TODO: Current preprocessing can be model dependent. This could be moved to inference time as well
            # Convert class-based method calls to function calls
            instance = process_multi_turn_test_case(instance, repo_root_dir)

            # Convert function calls to tools format and add them to the system prompt
            if "function" in instance:
                # Add the tools to the system prompt
                instance["function"] = func_doc_language_specific_pre_processing(instance["function"], test_category)
                instance["tools"] = convert_to_tool(instance["function"])

            f_out.write(json.dumps(instance) + "\n")


def download_and_process_bfcl_data(repo_url, subfolder_path, output_dir, file_prefix="BFCL_v3", model_type="nemotron"):
    """
    Download JSON files from the BFCL GitHub repo via cloning

    Args:
        repo_url: GitHub repository URL
        subfolder_path: Path to the data subfolder in case of BFCL
        output_dir: Directory to save the processed JSONL files
        file_prefix: Only process files starting with this prefix
        model_type: Formatting of functions and tools can be model dependent.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Clone repository with minimal depth
            LOG.info(f"Cloning repository {repo_url} to {temp_dir}")
            # v1.3 corresponds the release version for BFCL v3
            subprocess.run(
                ["git", "clone", "-b", "v1.3", "--depth=1", repo_url, temp_dir], check=True, capture_output=True
            )

            # Find the target folder
            target_folder = Path(temp_dir) / subfolder_path

            if not os.path.exists(target_folder):
                LOG.error(f"Folder {subfolder_path} not found in repository")
                raise FileNotFoundError(
                    f"Folder {subfolder_path} not found in {repo_url} cloned to {temp_dir}. The structure of BFCL has changed!"
                )

            # Find JSON files matching criteria
            json_pattern = os.path.join(target_folder, f"{file_prefix}*.json")
            json_files = glob.glob(json_pattern)

            LOG.info(f"Found {len(json_files)} JSON files matching pattern")

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            processed_files = 0
            for input_file in json_files:
                filename = os.path.basename(input_file)
                split_dirname = os.path.join(output_dir, filename.lstrip("BFCL_v3_").replace(".json", ""))
                if not os.path.exists(split_dirname):
                    os.makedirs(split_dirname)

                with open(os.path.join(split_dirname, "__init__.py"), "w") as f:
                    f.write(DEFAULT_SETTINGS)

                output_file = os.path.join(split_dirname, "test.jsonl")
                process_file(temp_dir, input_file, output_file, model_type=model_type)

                # Copy the original json file to the split directory
                shutil.copy(input_file, os.path.join(split_dirname, filename))
                processed_files += 1

            LOG.info(f"Successfully processed {processed_files} JSON files to {output_dir}")

        except subprocess.CalledProcessError as e:
            LOG.error(f"Git command failed: {e}")
            LOG.error("Make sure git is installed and the repository URL is correct")


def main(args):
    LOG.warning(
        "Currently processing according to the OpenAI model style which works for most models, including Qwen/Llama-Nemotron/DeepSeek."
    )

    download_and_process_bfcl_data(
        REPO_URL, DATA_FOLDER_PATH, output_dir=os.path.join(os.path.dirname(__file__)), model_type=args.model_type
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default=None, required=False)
    args = parser.parse_args()

    main(args)
