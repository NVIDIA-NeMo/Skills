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

"""Validate Ray-executor smoke run: assert the marker files exist + are well-formed.

Ground truth check for the Ray executor. If this passes, the executor
successfully:
  - routed via get_executor() Ray branch
  - submitted a job via RayJobClient
  - the job ran on the Ray cluster
  - the worker had shared-FS visibility into /workspace
  - dependency chaining placed THIS check_results step after the smoke task
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # for utils.py
from utils import assert_all, load_json, soft_assert  # noqa: E402


def check_results(workspace: str):
    marker_dir = os.path.join(workspace, "ray_smoke_output")

    soft_assert(
        os.path.isdir(marker_dir),
        f"expected marker directory at {marker_dir} (Ray worker did not write to /workspace)",
    )

    done_marker = os.path.join(marker_dir, "done.marker")
    soft_assert(
        os.path.isfile(done_marker),
        f"expected done.marker at {done_marker} — Ray job did not complete its entrypoint",
    )

    if os.path.isfile(done_marker):
        with open(done_marker, "rt", encoding="utf-8") as f:
            content = f.read().strip()
        soft_assert(
            content == "ray_executor_smoke ok",
            f"done.marker content unexpected: {content!r}",
        )

    timestamp_file = os.path.join(marker_dir, "timestamp.txt")
    soft_assert(
        os.path.isfile(timestamp_file),
        f"expected timestamp.txt at {timestamp_file}",
    )

    status_file = os.path.join(marker_dir, "status.json")
    soft_assert(
        os.path.isfile(status_file),
        f"expected status.json at {status_file}",
    )

    if os.path.isfile(status_file):
        # Narrow scope: json.load() raises json.JSONDecodeError (a ValueError
        # subclass) on bad JSON, OSError on file-read issues, and UnicodeDecodeError
        # (also ValueError) on non-UTF-8 content. Anything else is a bug worth
        # surfacing with the original traceback.
        try:
            status = load_json(status_file)
        except (OSError, ValueError) as exc:
            soft_assert(False, f"status.json could not be parsed as JSON: {exc}")
            status = None

        if status is not None:
            soft_assert(
                "host" in status and bool(status["host"]),
                "status.json missing or has empty 'host' (Ray worker did not record hostname)",
            )
            soft_assert(
                "python" in status and bool(status["python"]),
                "status.json missing or has empty 'python' (Ray worker did not record sys.version)",
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", required=True, help="Workspace directory containing ray smoke output")
    args = ap.parse_args()

    check_results(args.workspace)
    assert_all()


if __name__ == "__main__":
    main()
