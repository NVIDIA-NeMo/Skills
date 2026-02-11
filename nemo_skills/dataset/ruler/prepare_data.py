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
import json
import subprocess
import tempfile
from pathlib import Path

try:
    from nemo_skills.dataset.ruler.prepare_common import build_prepare_parser, parse_args_and_prepare_args
except ModuleNotFoundError:
    from prepare_common import build_prepare_parser, parse_args_and_prepare_args


def prepare_task_data_for_ns(task, data_dir, setup, data_format):
    """Resave data_dir/task/test.jsonl into current folder/task/test.jsonl."""
    original_path = Path(data_dir) / task / "test.jsonl"
    new_path = Path(__file__).parent / setup / task / "test.jsonl"
    Path(new_path).parent.mkdir(parents=True, exist_ok=True)
    with open(original_path, "r", encoding="utf-8") as fin, open(new_path, "w", encoding="utf-8") as fout:
        for line in fin:
            original_entry = json.loads(line)
            new_entry = {
                "index": original_entry["index"],
                "question": original_entry["input"],
                "expected_answer": original_entry["outputs"],
                "length": original_entry["length"],
            }
            if data_format == "default":
                new_entry["generation"] = original_entry["answer_prefix"].strip()
            elif data_format == "base":
                new_entry["generation"] = "\n" + original_entry["answer_prefix"].strip()
            fout.write(json.dumps(new_entry) + "\n")


def get_ruler_data(tasks, setup, template_tokens, max_seq_length, data_format, ruler_prepare_args, tmp_data_dir=None):
    if "cwe" in tasks:
        # checking if git-lfs is installed
        try:
            subprocess.run(
                ["git", "lfs", "--version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            print("Git LFS is not installed. Please install it to prepare 'cwe' ruler task")
            exit(1)

    # 1. installing necessary packages
    subprocess.run(["pip install wonderwords html2text tenacity"], check=True, shell=True)

    # 2. use provided tmp_data_dir or create a temporary directory
    if tmp_data_dir is not None:
        tmpdirname = tmp_data_dir
        Path(tmpdirname).mkdir(parents=True, exist_ok=True)
        tmpdir_context = None
    else:
        tmpdir_context = tempfile.TemporaryDirectory()
        tmpdirname = tmpdir_context.__enter__()

    try:
        json_dir = Path(tmpdirname) / "RULER" / "scripts" / "data" / "synthetic" / "json"
        required_files = [
            "english_words.json",
            "hotpotqa.json",
            "PaulGrahamEssays.json",
            "squad.json",
        ]
        # Check if all required files exist
        files_exist = all((json_dir / fname).exists() for fname in required_files)
        if not files_exist:
            subprocess.run(
                "git clone https://github.com/NVIDIA/RULER && "
                "cd RULER/scripts/data/synthetic/json && "
                "python download_paulgraham_essay.py && bash download_qa_dataset.sh",
                check=True,
                shell=True,
                cwd=tmpdirname,
            )

        max_seq_length -= template_tokens  # Adjusting for template tokens

        # preparing the datasets based on user options, in parallel
        def prepare_task(task):
            subprocess.run(
                f"python prepare.py --save_dir {tmpdirname}/ruler_data --benchmark synthetic "
                f"    --subset test --task {task} --tokenizer_type hf --model_template_type base --prepare_for_ns "
                f"    --num_samples 100 --max_seq_length {max_seq_length} {ruler_prepare_args}",
                shell=True,
                check=True,
                cwd=Path(tmpdirname) / "RULER" / "scripts" / "data",
            )

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(prepare_task, task) for task in tasks]
            for future in concurrent.futures.as_completed(futures):
                future.result()  # Will raise exception if any subprocess fails

        # resaving the data for nemo-skills format
        for task in tasks:
            prepare_task_data_for_ns(task, Path(tmpdirname) / "ruler_data", setup, data_format=data_format)

    finally:
        if tmpdir_context is not None:
            tmpdir_context.__exit__(None, None, None)


def main():
    parser = build_prepare_parser(description="Prepare RULER dataset data.")
    args, ruler_prepare_args = parse_args_and_prepare_args(parser)

    print(
        f"Preparing RULER dataset data for tasks: {args.tasks}, "
        f"data_format: {args.data_format}, "
        f"additional arguments: {ruler_prepare_args}"
    )
    get_ruler_data(
        args.tasks,
        args.setup,
        args.template_tokens,
        args.max_seq_length,
        args.data_format,
        ruler_prepare_args,
        tmp_data_dir=args.tmp_data_dir,
    )
    print("RULER data preparation completed.")


if __name__ == "__main__":
    main()
