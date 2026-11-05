# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Prepare SPEED-Bench using the shared implementation in NeMo Gym."""

import argparse
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

# Pin the standalone Gym script so benchmark preparation is reproducible.
GYM_COMMIT = "9022d42b1bf3679c3bee568ba7e63306e0e91324"
PREPARE_URL = f"https://raw.githubusercontent.com/NVIDIA-NeMo/Gym/{GYM_COMMIT}/benchmarks/speed-bench/prepare.py"


def prepare_data(args: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        script_path = Path(tmp_dir) / "prepare.py"
        urllib.request.urlretrieve(PREPARE_URL, script_path)
        subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--config",
                args.config,
                "--format",
                "raw",
                "--output-dir",
                str(args.output_dir),
            ],
            check=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare SPEED-Bench for NeMo Skills using NeMo Gym.")
    parser.add_argument(
        "--config",
        default="all",
        help='SPEED-Bench configuration to prepare, or "all" (default: %(default)s). Validated by Gym.',
    )
    parser.add_argument(
        "--output_dir",
        "--output-dir",
        type=Path,
        default=Path(__file__).parent,
        help="Directory to save the prepared dataset files (default: %(default)s)",
    )
    prepare_data(parser.parse_args())
