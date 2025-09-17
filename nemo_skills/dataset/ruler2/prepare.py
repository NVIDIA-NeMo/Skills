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


import argparse
import concurrent.futures
import json
import subprocess
from pathlib import Path

DEFAULT_SETTINGS = """
DATASET_GROUP = "long-context"
METRICS_TYPE = "ruler2"
EVAL_ARGS = "{eval_args}"
GENERATION_ARGS = (
    "++prompt_config=generic/default "
)
"""


def prepare_mk_niah_basic(output_folder, tokenizer_type, tokenizer_path, length, dataset_size):
    subprocess.run(
        f"python -m nemo_skills.dataset.ruler2.prepare_niah "
        f"--output_folder {output_folder} "
        f"--tokenizer_type ${tokenizer_type} "
        f"--tokenizer_path ${tokenizer_path} "
        f"--max_seq_length ${length} "
        f"--num_samples ${dataset_size} "
        f"--random_seed 42 "
        f"--num_needle_k 1 "
        f"--num_needle_v 1 "
        f"--num_needle_q 1 "
        f"--type_haystack needle "
        f"--type_needle_k words "
        f"--type_needle_v numbers "
        f"--num_digits_v 10",
        shell=True,
        check=True,
    )

def prepare_mk_niah_easy(output_folder, tokenizer_type, tokenizer_path, length, dataset_size):
    subprocess.run(
        f"python -m nemo_skills.dataset.ruler2.prepare_mmlu "
        f"--output_folder {output_folder} "
        f"--tokenizer_type ${tokenizer_type} "
        f"--tokenizer_path ${tokenizer_path} "
        f"--max_seq_length ${length} "
        f"--num_samples ${dataset_size} "
        f"--random_seed 42 "
        f"--dataset mmlu "
        f"--fewshot 0 "
        f"--prompt_type instruct "
        f"--num_order 0 "
        f"--task_type retrieve "
        f"--algo_type single",
        shell=True,
        check=True,
    )

def prepare_mk_niah_medium(output_folder, tokenizer_type, tokenizer_path, length, dataset_size):
    subprocess.run(
        f"python -m nemo_skills.dataset.ruler2.prepare_mmlu "
        f"--output_folder {output_folder} "
        f"--tokenizer_type ${tokenizer_type} "
        f"--tokenizer_path ${tokenizer_path} "
        f"--max_seq_length ${length} "
        f"--num_samples ${dataset_size} "
        f"--random_seed 42 "
        f"--dataset mmlu "
        f"--fewshot 5 "
        f"--prompt_type instruct "
        f"--num_order 0 "
        f"--task_type solve "
        f"--algo_type 2steps",
        shell=True,
        check=True,
    )

def prepare_mk_niah_hard(output_folder, tokenizer_type, tokenizer_path, length, dataset_size):
    subprocess.run(
        f"python -m nemo_skills.dataset.ruler2.prepare_mmlu "
        f"--output_folder {output_folder} "
        f"--tokenizer_type ${tokenizer_type} "
        f"--tokenizer_path ${tokenizer_path} "
        f"--max_seq_length ${length} "
        f"--num_samples ${dataset_size} "
        f"--random_seed 42 "
        f"--dataset mmlu "
        f"--fewshot 5 "
        f"--prompt_type instruct "
        f"--num_order 0 "
        f"--task_type solve "
        f"--algo_type single",
        shell=True,
        check=True,
    )

def prepare_mv_niah_basic(output_folder, tokenizer_type, tokenizer_path, length, dataset_size):
    subprocess.run(
        f"python -m prepare.py nemo_skills.dataset.ruler2.prepare_niah "
        f"--output_folder {output_folder} "
        f"--tokenizer_type ${tokenizer_type} "
        f"--tokenizer_path ${tokenizer_path} "
        f"--max_seq_length ${length} "
        f"--num_samples ${dataset_size} "
        f"--random_seed 42 "
        f"--num_needle_k 1 "
        f"--num_needle_v 4 "
        f"--num_needle_q 1 "
        f"--type_haystack needle "
        f"--type_needle_k words "
        f"--type_needle_v numbers "
        f"--num_digits_v 10",
        shell=True,
        check=True,
    )

def prepare_mv_niah_easy(output_folder, tokenizer_type, tokenizer_path, length, dataset_size):
    subprocess.run(
        f"python -m nemo_skills.dataset.ruler2.prepare_mmlu "
        f"--output_folder {output_folder} "
        f"--tokenizer_type ${tokenizer_type} "
        f"--tokenizer_path ${tokenizer_path} "
        f"--max_seq_length ${length} "
        f"--num_samples ${dataset_size} "
        f"--random_seed 42 "
        f"--dataset mmlu "
        f"--fewshot 0 "
        f"--prompt_type instruct "
        f"--num_order 4 "
        f"--task_type niah "
        f"--algo_type single",
        shell=True,
        check=True,
    )

def prepare_mv_niah_medium(output_folder, tokenizer_type, tokenizer_path, length, dataset_size):
    subprocess.run(
        f"python -m nemo_skills.dataset.ruler2.prepare_mmlu "
        f"--output_folder {output_folder} "
        f"--tokenizer_type ${tokenizer_type} "
        f"--tokenizer_path ${tokenizer_path} "
        f"--max_seq_length ${length} "
        f"--num_samples ${dataset_size} "
        f"--random_seed 42 "
        f"--dataset mmlu "
        f"--fewshot 0 "
        f"--prompt_type instruct "
        f"--num_order 4 "
        f"--task_type retrieve "
        f"--algo_type 2steps",
        shell=True,
        check=True,
    )

def prepare_mv_niah_hard(output_folder, tokenizer_type, tokenizer_path, length, dataset_size):
    subprocess.run(
        f"python -m nemo_skills.dataset.ruler2.prepare_mmlu "
        f"--output_folder {output_folder} "
        f"--tokenizer_type ${tokenizer_type} "
        f"--tokenizer_path ${tokenizer_path} "
        f"--max_seq_length ${length} "
        f"--num_samples ${dataset_size} "
        f"--random_seed 42 "
        f"--dataset mmlu "
        f"--fewshot 0 "
        f"--prompt_type instruct "
        f"--num_order 4 "
        f"--task_type retrieve "
        f"--algo_type single",
        shell=True,
        check=True,
    )


def prepare_qa_basic(output_folder, tokenizer_type, tokenizer_path, length, dataset_size):
    subprocess.run(
        f"python -m nemo_skills.dataset.ruler2.prepare_qa "
        f"--output_folder {output_folder} "
        f"--tokenizer_type ${tokenizer_type} "
        f"--tokenizer_path ${tokenizer_path} "
        f"--max_seq_length ${length} "
        f"--num_samples ${dataset_size} "
        f"--random_seed 42 "
        f"--dataset hotpotqa "
        f"--fewshot 0 "
        f"--prompt_type instruct "
        f"--task_type retrieve "
        f"--query_type doc",
        shell=True,
        check=True,
    )

def prepare_qa_easy(output_folder, tokenizer_type, tokenizer_path, length, dataset_size):
    subprocess.run(
        f"python -m nemo_skills.dataset.ruler2.prepare_qa "
        f"--output_folder {output_folder} "
        f"--tokenizer_type ${tokenizer_type} "
        f"--tokenizer_path ${tokenizer_path} "
        f"--max_seq_length ${length} "
        f"--num_samples ${dataset_size} "
        f"--random_seed 42 "
        f"--dataset hotpotqa "
        f"--fewshot 0 "
        f"--prompt_type instruct "
        f"--task_type retrieve "
        f"--query_type question",
        shell=True,
        check=True,
    )


def prepare_qa_medium(output_folder, tokenizer_type, tokenizer_path, length, dataset_size):
    subprocess.run(
        f"python -m nemo_skills.dataset.ruler2.prepare_qa "
        f"--output_folder {output_folder} "
        f"--tokenizer_type ${tokenizer_type} "
        f"--tokenizer_path ${tokenizer_path} "
        f"--max_seq_length ${length} "
        f"--num_samples ${dataset_size} "
        f"--random_seed 42 "
        f"--dataset hotpotqa "
        f"--fewshot 0 "
        f"--prompt_type instruct "
        f"--task_type solve "
        f"--algo_type 2steps",
        shell=True,
        check=True,
    )

def prepare_qa_hard(output_folder, tokenizer_type, tokenizer_path, length, dataset_size):
    subprocess.run(
        f"python -m nemo_skills.dataset.ruler2.prepare_qa "
        f"--output_folder {output_folder} "
        f"--tokenizer_type ${tokenizer_type} "
        f"--tokenizer_path ${tokenizer_path} "
        f"--max_seq_length ${length} "
        f"--num_samples ${dataset_size} "
        f"--random_seed 42 "
        f"--dataset hotpotqa "
        f"--fewshot 0 "
        f"--prompt_type instruct "
        f"--task_type solve "
        f"--algo_type single",
        shell=True,
        check=True,
    )



def prepare_task_for_ns(output_folder):
    """Adding proper __init__.py"""
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    with open(output_folder / "__init__.py", "w", encoding="utf-8") as init_file:
        if task in ["mk_niah_medium", "mk_niah_hard"]:
            eval_args = "++eval_type=multichoice"
        elif task in ["mv_niah_medium"]:
            eval_args = "++eval_type=ruler2 ++eval_config.match_type=2steps"
        elif "qa" in task:
            eval_args = "++eval_type=ruler2 ++eval_config.match_type=part"
        else:
            eval_args = "++eval_type=ruler2 ++eval_config.match_type=all"

        init_file.write(DEFAULT_SETTINGS.format(eval_args=eval_args))

def prepare_dataset(tasks, setup, max_seq_length, tokenizer_type, tokenizer_path, dataset_size):
    prepare_task = {
        "mk_niah_basic": prepare_mk_niah_basic,
        "mk_niah_easy": prepare_mk_niah_easy,
        "mk_niah_medium": prepare_mk_niah_medium,
        "mk_niah_hard": prepare_mk_niah_hard,
        "mv_niah_basic": prepare_mv_niah_basic,
        "mv_niah_easy": prepare_mv_niah_easy,
        "mv_niah_medium": prepare_mv_niah_medium,
        "mv_niah_hard": prepare_mv_niah_hard,
        "qa_basic": prepare_qa_basic,
        "qa_easy": prepare_qa_easy,
        "qa_medium": prepare_qa_medium,
        "qa_hard": prepare_qa_hard,
    }


    output_folder = Path(__file__).parent / setup

    # 1. installing necessary packages
    subprocess.run(["pip install wonderwords html2text tenacity"], check=True, shell=True)

    for task in tasks:
        prepare_task_for_ns(output_folder / task)

    # preparing the datasets based on user options, in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(prepare_task[task], 
            str(output_folder / task), 
            tokenizer_type, 
            tokenizer_path, 
            max_seq_length, 
            dataset_size
        ) for task in tasks]
        for future in concurrent.futures.as_completed(futures):
            future.result()  # Will raise exception if any subprocess fails

    with open(output_folder / "__init__.py", "w", encoding="utf-8") as init_file:
        init_file.write("IS_BENCHMARK_GROUP = True\n")
        init_file.write("SCORE_MODULE = 'nemo_skills.dataset.ruler2.ruler2_score'\n")
        benchmarks = ", ".join(f"'ruler2.{setup}.{task}': {{}}" for task in tasks)
        init_file.write(f"BENCHMARKS = {{{benchmarks}}}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare RULER2 dataset.")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=[
            "mk_niah_basic",
            "mk_niah_easy",
            "mk_niah_medium",
            "mk_niah_hard",
            "mv_niah_basic",
            "mv_niah_easy",
            "mv_niah_medium",
            "mv_niah_hard",
            "qa_basic",
            "qa_easy",
            "qa_medium",
            "qa_hard",
        ],
        help="List of tasks to prepare for RULER2 dataset.",
    )
    parser.add_argument(
        "--setup",
        type=str,
        required=True,
        help="Name of the setup for RULER2 dataset. Typically should be <model_name>_<sequence_length>.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        required=True,
        help="Sequence length to check with RULER2.",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="hf",
        help="Type of the tokenizer to use.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        required=True,
        help="Path to the tokenizer to use.",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=100,
        help="Number of samples to prepare for RULER2 dataset.",
    )

    args, unknown = parser.parse_known_args()
    prepare_dataset(
        args.tasks,
        args.setup,
        args.max_seq_length,
        args.tokenizer_type,
        args.tokenizer_path,
        args.dataset_size,
    )
    print("RULER2 dataset preparation completed.")