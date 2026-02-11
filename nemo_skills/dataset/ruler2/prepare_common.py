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

DEFAULT_TASKS = [
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
]


def build_prepare_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=DEFAULT_TASKS,
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
    return parser


def parse_known_args(parser: argparse.ArgumentParser):
    args, unknown = parser.parse_known_args()
    return args, unknown
