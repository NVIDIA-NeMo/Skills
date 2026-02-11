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

import concurrent.futures
import subprocess
from pathlib import Path

try:
    from nemo_skills.dataset.ruler2.prepare_common import build_prepare_parser, parse_known_args
except ModuleNotFoundError:
    from prepare_common import build_prepare_parser, parse_known_args


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


if __name__ == "__main__":
    parser = build_prepare_parser(description="Prepare RULER2 dataset data.")
    args, unknown = parse_known_args(parser)
    _ = unknown
    prepare_dataset(
        args.tasks,
        args.setup,
        args.max_seq_length,
        args.tokenizer_type,
        args.tokenizer_path,
        args.dataset_size,
    )
    print("RULER2 data preparation completed.")
