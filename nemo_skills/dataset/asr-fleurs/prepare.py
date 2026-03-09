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

"""Prepare FLEURS dataset for ASR evaluation.

Reads a NeMo-style manifest (JSONL with audio_filepath, text, lang) and
converts it to the nemo-skills evaluation format.

Audio files are expected to already exist on disk — this script does NOT
download or copy them.  It reads WAV headers to get duration and writes
audio paths that will be valid inside the evaluation container.

Usage:
    python prepare.py \\
        --manifest_path /path/to/filtered_manifest.json \\
        --audio_root /path/to/filtered_fleurs_data \\
        --output_dir /dataset/asr-fleurs

    The manifest's audio_filepath values (e.g. "fleurs_data/en_us/test/x.wav")
    have their first path component stripped and are resolved relative to
    --audio_root.  So "fleurs_data/en_us/test/x.wav" becomes
    "{audio_root}/en_us/test/x.wav" on disk and
    "/dataset/asr-fleurs/data/en_us/test/x.wav" in the output JSONL.
"""

import argparse
import json
from pathlib import Path

import soundfile as sf

SYSTEM_MESSAGE = "You are a helpful assistant. /no_think"
MIN_AUDIO_DURATION = 0.1


def remap_audio_path(manifest_path: str) -> str:
    """Strip the first path component from the manifest's audio_filepath.

    "fleurs_data/en_us/test/en_us_0000.wav" -> "en_us/test/en_us_0000.wav"
    """
    parts = Path(manifest_path).parts
    return str(Path(*parts[1:])) if len(parts) > 1 else manifest_path


def format_entry(text, audio_container_path, duration, subset, sample_id):
    """Build a single nemo-skills JSONL entry."""
    system_message = {"role": "system", "content": SYSTEM_MESSAGE}
    user_message = {
        "role": "user",
        "content": "Transcribe the following audio.",
        "audio": {
            "path": audio_container_path,
            "duration": float(duration),
        },
    }
    return {
        "task_type": "ASR",
        "expected_answer": text,
        "messages": [system_message, user_message],
        "subset_for_metrics": subset,
        "id": sample_id,
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare FLEURS data for nemo-skills ASR evaluation")
    parser.add_argument(
        "--manifest_path",
        required=True,
        help="Path to NeMo manifest JSONL (e.g. filtered_manifest.json)",
    )
    parser.add_argument(
        "--audio_root",
        required=True,
        help="Root directory containing the actual audio files "
        "(e.g. /lustre/.../filtered_fleurs_data). Manifest paths are "
        "resolved relative to this after stripping the first component.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to write test.jsonl into (default: /dataset/asr-fleurs "
        "if it exists, otherwise next to this script).",
    )
    parser.add_argument(
        "--container_data_prefix",
        default="/dataset/asr-fleurs/data",
        help="Path prefix used in JSONL audio entries — must match the "
        "container mount point at eval time (default: /dataset/asr-fleurs/data).",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest_path)
    audio_root = Path(args.audio_root)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        default = Path("/dataset/asr-fleurs")
        output_dir = default if default.exists() else Path(__file__).parent

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "test.jsonl"

    print(f"Manifest:  {manifest_path}")
    print(f"Audio root: {audio_root}")
    print(f"Output:    {output_file}")
    print(f"Container prefix: {args.container_data_prefix}")

    count = 0
    skipped = 0
    missing = 0

    with open(manifest_path, encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
        for line_num, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue

            entry = json.loads(line)
            text = entry.get("text", "").strip()
            if not text:
                skipped += 1
                continue

            manifest_audio = entry["audio_filepath"]
            relative_path = remap_audio_path(manifest_audio)

            disk_path = audio_root / relative_path
            if not disk_path.exists():
                missing += 1
                if missing <= 5:
                    print(f"  Warning: audio not found: {disk_path}")
                continue

            info = sf.info(str(disk_path))
            duration = info.duration
            if duration < MIN_AUDIO_DURATION:
                skipped += 1
                continue

            container_path = f"{args.container_data_prefix}/{relative_path}"
            lang = entry.get("lang", "en_us")
            subset = f"fleurs_{lang}"
            sample_id = Path(relative_path).stem

            formatted = format_entry(text, container_path, duration, subset, sample_id)
            fout.write(json.dumps(formatted) + "\n")
            count += 1

    print(f"\nWrote {count} samples to {output_file}")
    if skipped:
        print(f"Skipped {skipped} samples (empty text or audio < {MIN_AUDIO_DURATION}s)")
    if missing:
        print(f"Missing {missing} audio files (first 5 warnings shown above)")


if __name__ == "__main__":
    main()
