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
    "niah_single_1",
    "niah_single_2",
    "niah_single_3",
    "niah_multikey_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "niah_multivalue",
    "niah_multiquery",
    "vt",
    "cwe",
    "fwe",
    "qa_1",
    "qa_2",
]

MISSING_RULER_ARGS_MESSAGE = (
    "ERROR: Can't prepare ruler without arguments provided! "
    "Skipping the preparation step.\n"
    "Example ruler prepare command:\n"
    "ns prepare_data ruler --setup llama_128k "
    "--tokenizer_path meta-llama/Llama-3.1-8B-Instruct --max_seq_length 131072"
)


def build_prepare_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=DEFAULT_TASKS,
        help="List of tasks to prepare for RULER dataset.",
    )
    parser.add_argument(
        "--setup",
        type=str,
        required=True,
        help="Name of the setup for RULER dataset. Typically should be <model_name>_<sequence_length>.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        required=True,
        help="Sequence length to check with RULER.",
    )
    parser.add_argument(
        "--template_tokens",
        type=int,
        default=50,
        help="Number of tokens in chat template.",
    )
    parser.add_argument(
        "--tmp_data_dir",
        type=str,
        default=None,
        help="Directory to store intermediate data.",
    )
    parser.add_argument(
        "--data_format",
        type=str,
        default="default",
        choices=["default", "base", "chat"],
        help="""
        default: use default format, answer_prefix is added in the generation field.
        base: use base format, answer_prefix is added in the generation field with a newline separator.
        chat: use chat format, answer_prefix is removed.
        """,
    )
    return parser


def parse_args_and_prepare_args(parser: argparse.ArgumentParser):
    args, unknown = parser.parse_known_args()
    ruler_prepare_args = " ".join(unknown)
    if not ruler_prepare_args:
        print(MISSING_RULER_ARGS_MESSAGE)
        raise SystemExit(0)
    return args, ruler_prepare_args
