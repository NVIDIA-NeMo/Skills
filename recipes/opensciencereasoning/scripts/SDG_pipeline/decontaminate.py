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
import json
import logging


def main():
    """Filter the input dataset based on contamination check results.

    Reads contamination decisions from `--dec_path` (expects a JSONL with
    fields `problem` and boolean `contaminated`). Keeps only problems marked
    non-contaminated, writing the filtered set to `--save_path`.

    If `--with_duplicates` is False, repeated problems are emitted once.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(
        description="Decontaminate the input file. Reads contamination decisions and filters the input JSONL."
    )
    parser.add_argument(
        "-i",
        "--input_path",
        required=True,
        help="Path to input JSONL to be filtered (original dataset)",
    )
    parser.add_argument(
        "-d",
        "--dec_path",
        required=True,
        help="Path to JSONL with contamination decisions ('problem' and boolean 'contaminated')",
    )
    parser.add_argument(
        "-s",
        "--save_path",
        required=True,
        help="Path to write filtered JSONL (non-contaminated problems)",
    )
    parser.add_argument(
        "-f",
        "--with_duplicates",
        default=False,
        help="If set, allow duplicate problems in output; otherwise emit each once",
    )
    args = parser.parse_args()

    dec = set()
    logging.info(f"Reading contamination decisions from {args.dec_path}")

    with open(args.dec_path) as fin:
        for line in fin:
            sample = json.loads(line)
            if not sample:
                logging.warning("Skipping empty line")
                continue
            if not sample["contaminated"]:
                dec.add(sample["problem"])

    with open(args.input_path) as fin, open(args.save_path, "w") as fout:
        for line in fin:
            sample = json.loads(line)
            if sample["problem"] in dec:
                if not args.with_duplicates:
                    dec.remove(sample["problem"])
                fout.write(line)


if __name__ == "__main__":
    main()
