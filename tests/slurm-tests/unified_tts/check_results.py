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

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))  # for utils.py
from utils import assert_all, soft_assert  # noqa: E402

EXPECTED_NUM_SAMPLES = 6


def load_outputs(output_dir: Path) -> list[dict]:
    rows: list[dict] = []
    files = sorted(output_dir.glob("output*.jsonl"))
    soft_assert(len(files) > 0, f"No output JSONL files found in {output_dir}")
    for fpath in files:
        with fpath.open("rt", encoding="utf-8") as fin:
            for line in fin:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def resolve_audio_path(audio_path: str, workspace: str) -> Path:
    path = Path(audio_path)
    if path.is_absolute():
        return path
    return Path(workspace) / "tts_outputs" / "audio" / path.name


def check_tts_results(workspace: str):
    output_dir = Path(workspace) / "tts_outputs"
    soft_assert(output_dir.exists(), f"Missing output directory: {output_dir}")
    if not output_dir.exists():
        return

    rows = load_outputs(output_dir)
    soft_assert(len(rows) == EXPECTED_NUM_SAMPLES, f"Expected {EXPECTED_NUM_SAMPLES} outputs, found {len(rows)}")

    for row in rows:
        sample_id = row.get("id", "<missing-id>")
        audio_info = row.get("audio")
        soft_assert(isinstance(audio_info, dict), f"Missing 'audio' block in output row {sample_id}")
        if not isinstance(audio_info, dict):
            continue

        audio_path = audio_info.get("path")
        soft_assert(bool(audio_path), f"Missing audio path for row {sample_id}")
        if not audio_path:
            continue

        resolved = resolve_audio_path(audio_path, workspace)
        soft_assert(resolved.exists(), f"Audio file does not exist for {sample_id}: {resolved}")
        if not resolved.exists():
            continue

        size_bytes = resolved.stat().st_size
        soft_assert(size_bytes > 0, f"Audio file is empty for {sample_id}: {resolved}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, help="Workspace directory containing results")
    args = parser.parse_args()

    check_tts_results(args.workspace)
    assert_all()


if __name__ == "__main__":
    main()
