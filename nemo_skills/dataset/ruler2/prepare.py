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
import subprocess
from pathlib import Path

DEFAULT_SETTINGS = """
DATASET_GROUP = "long-context"
METRICS_TYPE = "{metrics_type}"
GENERATION_ARGS = (
    "++prompt_config=generic/default "
    "{eval_args} "
)
"""


def prepare_mk_niah_basic(output_folder, tokenizer_type, tokenizer_path, length, dataset_size):
    subprocess.run(
        [
            "python",
            "-m",
            "nemo_skills.dataset.ruler2.prepare_niah",
            "--output_folder",
            output_folder,
            "--tokenizer_type",
            tokenizer_type,
            "--tokenizer_path",
            tokenizer_path,
            "--max_seq_length",
            str(length),
            "--num_samples",
            str(dataset_size),
            "--random_seed",
            "42",
            "--num_needle_k",
            "1",
            "--num_needle_v",
            "1",
            "--num_needle_q",
            "1",
            "--type_haystack",
            "needle",
            "--type_needle_k",
            "words",
            "--type_needle_v",
            "numbers",
            "--num_digits_v",
            "10",
        ],
        check=True,
    )


def prepare_mk_niah_easy(output_folder, tokenizer_type, tokenizer_path, length, dataset_size):
    subprocess.run(
        [
            "python",
            "-m",
            "nemo_skills.dataset.ruler2.prepare_mmlu",
            "--output_folder",
            output_folder,
            "--tokenizer_type",
            tokenizer_type,
            "--tokenizer_path",
            tokenizer_path,
            "--max_seq_length",
            str(length),
            "--num_samples",
            str(dataset_size),
            "--random_seed",
            "42",
            "--dataset",
            "mmlu",
            "--fewshot",
            "0",
            "--prompt_type",
            "instruct",
            "--num_order",
            "0",
            "--task_type",
            "retrieve",
            "--algo_type",
            "single",
        ],
        check=True,
    )


def prepare_mk_niah_medium(output_folder, tokenizer_type, tokenizer_path, length, dataset_size):
    subprocess.run(
        [
            "python",
            "-m",
            "nemo_skills.dataset.ruler2.prepare_mmlu",
            "--output_folder",
            output_folder,
            "--tokenizer_type",
            tokenizer_type,
            "--tokenizer_path",
            tokenizer_path,
            "--max_seq_length",
            str(length),
            "--num_samples",
            str(dataset_size),
            "--random_seed",
            "42",
            "--dataset",
            "mmlu",
            "--fewshot",
            "5",
            "--prompt_type",
            "instruct",
            "--num_order",
            "0",
            "--task_type",
            "solve",
            "--algo_type",
            "2steps",
        ],
        check=True,
    )


def prepare_mk_niah_hard(output_folder, tokenizer_type, tokenizer_path, length, dataset_size):
    subprocess.run(
        [
            "python",
            "-m",
            "nemo_skills.dataset.ruler2.prepare_mmlu",
            "--output_folder",
            output_folder,
            "--tokenizer_type",
            tokenizer_type,
            "--tokenizer_path",
            tokenizer_path,
            "--max_seq_length",
            str(length),
            "--num_samples",
            str(dataset_size),
            "--random_seed",
            "42",
            "--dataset",
            "mmlu",
            "--fewshot",
            "5",
            "--prompt_type",
            "instruct",
            "--num_order",
            "0",
            "--task_type",
            "solve",
            "--algo_type",
            "single",
        ],
        check=True,
    )


def prepare_mv_niah_basic(output_folder, tokenizer_type, tokenizer_path, length, dataset_size):
    subprocess.run(
        [
            "python",
            "-m",
            "nemo_skills.dataset.ruler2.prepare_niah",
            "--output_folder",
            output_folder,
            "--tokenizer_type",
            tokenizer_type,
            "--tokenizer_path",
            tokenizer_path,
            "--max_seq_length",
            str(length),
            "--num_samples",
            str(dataset_size),
            "--random_seed",
            "42",
            "--num_needle_k",
            "1",
            "--num_needle_v",
            "4",
            "--num_needle_q",
            "1",
            "--type_haystack",
            "needle",
            "--type_needle_k",
            "words",
            "--type_needle_v",
            "numbers",
            "--num_digits_v",
            "10",
        ],
        check=True,
    )


def prepare_mv_niah_easy(output_folder, tokenizer_type, tokenizer_path, length, dataset_size):
    subprocess.run(
        [
            "python",
            "-m",
            "nemo_skills.dataset.ruler2.prepare_mmlu",
            "--output_folder",
            output_folder,
            "--tokenizer_type",
            tokenizer_type,
            "--tokenizer_path",
            tokenizer_path,
            "--max_seq_length",
            str(length),
            "--num_samples",
            str(dataset_size),
            "--random_seed",
            "42",
            "--dataset",
            "mmlu",
            "--fewshot",
            "0",
            "--prompt_type",
            "instruct",
            "--num_order",
            "4",
            "--task_type",
            "niah",
            "--algo_type",
            "single",
        ],
        check=True,
    )


def prepare_mv_niah_medium(output_folder, tokenizer_type, tokenizer_path, length, dataset_size):
    subprocess.run(
        [
            "python",
            "-m",
            "nemo_skills.dataset.ruler2.prepare_mmlu",
            "--output_folder",
            output_folder,
            "--tokenizer_type",
            tokenizer_type,
            "--tokenizer_path",
            tokenizer_path,
            "--max_seq_length",
            str(length),
            "--num_samples",
            str(dataset_size),
            "--random_seed",
            "42",
            "--dataset",
            "mmlu",
            "--fewshot",
            "0",
            "--prompt_type",
            "instruct",
            "--num_order",
            "4",
            "--task_type",
            "retrieve",
            "--algo_type",
            "2steps",
        ],
        check=True,
    )


def prepare_mv_niah_hard(output_folder, tokenizer_type, tokenizer_path, length, dataset_size):
    subprocess.run(
        [
            "python",
            "-m",
            "nemo_skills.dataset.ruler2.prepare_mmlu",
            "--output_folder",
            output_folder,
            "--tokenizer_type",
            tokenizer_type,
            "--tokenizer_path",
            tokenizer_path,
            "--max_seq_length",
            str(length),
            "--num_samples",
            str(dataset_size),
            "--random_seed",
            "42",
            "--dataset",
            "mmlu",
            "--fewshot",
            "0",
            "--prompt_type",
            "instruct",
            "--num_order",
            "4",
            "--task_type",
            "retrieve",
            "--algo_type",
            "single",
        ],
        check=True,
    )


def prepare_qa_basic(output_folder, tokenizer_type, tokenizer_path, length, dataset_size):
    subprocess.run(
        [
            "python",
            "-m",
            "nemo_skills.dataset.ruler2.prepare_qa",
            "--output_folder",
            output_folder,
            "--tokenizer_type",
            tokenizer_type,
            "--tokenizer_path",
            tokenizer_path,
            "--max_seq_length",
            str(length),
            "--num_samples",
            str(dataset_size),
            "--random_seed",
            "42",
            "--dataset",
            "hotpotqa",
            "--fewshot",
            "0",
            "--prompt_type",
            "instruct",
            "--task_type",
            "retrieve",
            "--query_type",
            "doc",
        ],
        check=True,
    )


def prepare_qa_easy(output_folder, tokenizer_type, tokenizer_path, length, dataset_size):
    subprocess.run(
        [
            "python",
            "-m",
            "nemo_skills.dataset.ruler2.prepare_qa",
            "--output_folder",
            output_folder,
            "--tokenizer_type",
            tokenizer_type,
            "--tokenizer_path",
            tokenizer_path,
            "--max_seq_length",
            str(length),
            "--num_samples",
            str(dataset_size),
            "--random_seed",
            "42",
            "--dataset",
            "hotpotqa",
            "--fewshot",
            "0",
            "--prompt_type",
            "instruct",
            "--task_type",
            "retrieve",
            "--query_type",
            "question",
        ],
        check=True,
    )


def prepare_qa_medium(output_folder, tokenizer_type, tokenizer_path, length, dataset_size):
    subprocess.run(
        [
            "python",
            "-m",
            "nemo_skills.dataset.ruler2.prepare_qa",
            "--output_folder",
            output_folder,
            "--tokenizer_type",
            tokenizer_type,
            "--tokenizer_path",
            tokenizer_path,
            "--max_seq_length",
            str(length),
            "--num_samples",
            str(dataset_size),
            "--random_seed",
            "42",
            "--dataset",
            "hotpotqa",
            "--fewshot",
            "0",
            "--prompt_type",
            "instruct",
            "--task_type",
            "solve",
            "--algo_type",
            "2steps",
        ],
        check=True,
    )


def prepare_qa_hard(output_folder, tokenizer_type, tokenizer_path, length, dataset_size):
    subprocess.run(
        [
            "python",
            "-m",
            "nemo_skills.dataset.ruler2.prepare_qa",
            "--output_folder",
            output_folder,
            "--tokenizer_type",
            tokenizer_type,
            "--tokenizer_path",
            tokenizer_path,
            "--max_seq_length",
            str(length),
            "--num_samples",
            str(dataset_size),
            "--random_seed",
            "42",
            "--dataset",
            "hotpotqa",
            "--fewshot",
            "0",
            "--prompt_type",
            "instruct",
            "--task_type",
            "solve",
            "--algo_type",
            "single",
        ],
        check=True,
    )


def prepare_task_for_ns(output_folder, task):
    """Adding proper __init__.py"""
    output_folder = Path(output_folder) / task
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    with open(output_folder / "__init__.py", "w", encoding="utf-8") as init_file:
        if task in ["mk_niah_medium", "mk_niah_hard"]:
            metrics_type = "multichoice"
            eval_args = "++eval_type=multichoice"
        elif task in ["mv_niah_medium"]:
            metrics_type = "ruler2"
            eval_args = "++eval_type=ruler2 ++eval_config.match_type=2steps"
        elif "qa" in task:
            metrics_type = "ruler2"
            eval_args = "++eval_type=ruler2 ++eval_config.match_type=part"
        else:
            metrics_type = "ruler2"
            eval_args = "++eval_type=ruler2 ++eval_config.match_type=all"

        init_file.write(DEFAULT_SETTINGS.format(metrics_type=metrics_type, eval_args=eval_args))


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
    # subprocess.run(["pip", "install", "wonderwords", "html2text", "tenacity"], check=True)

    for task in tasks:
        prepare_task_for_ns(output_folder, task)

    # preparing the datasets based on user options, in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                prepare_task[task],
                str(output_folder / task),
                tokenizer_type,
                tokenizer_path,
                max_seq_length,
                dataset_size,
            )
            for task in tasks
        ]
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
