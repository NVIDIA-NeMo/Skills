# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Prepare AppTek Call-Center Dialogues for NeMo Skills ASR evaluation.

The upstream HuggingFace repository stores one metadata file and one audio
directory per accent. This script downloads the requested accents, maps all
rows into a single NeMo Skills ``test.jsonl`` split, and writes container-mount
audio paths into the manifest so cluster evaluation works without further
configuration.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

HF_REPO_ID = "apptek-com/apptek_callcenter_dialogues"
BENCHMARK_NAME = "apptek-callcenter-dialogues"
SYSTEM_MESSAGE = "You are a helpful assistant. /no_think"
PROMPT = "Transcribe the following audio."
DEFAULT_AUDIO_PREFIX = f"/data/{BENCHMARK_NAME}"

ACCENT_CODES = (
    "en-AU",
    "en-CA",
    "en-CN",
    "en-GB",
    "en-GB_SCT",
    "en-GB_WLS",
    "en-IE",
    "en-IN",
    "en-MX",
    "en-SG",
    "en-US_Aave",
    "en-US_General",
    "en-US_Southern",
    "en-ZA",
)


def _metadata_paths(data_dir: Path, accent_codes: Iterable[str]) -> list[Path]:
    """Return expected metadata paths for selected accents."""
    return [_metadata_path(data_dir, accent) for accent in accent_codes]


def _metadata_path(data_dir: Path, accent_code: str) -> Path:
    """Return the metadata path, supporting both old and current HF layouts."""
    direct_path = data_dir / accent_code / "metadata.jsonl"
    if direct_path.exists():
        return direct_path
    return data_dir / "test" / accent_code / "metadata.jsonl"


def _metadata_complete(data_dir: Path, accent_codes: Iterable[str]) -> bool:
    """Check whether all selected metadata files are already present."""
    return all(path.exists() for path in _metadata_paths(data_dir, accent_codes))


def _iter_metadata_rows(data_dir: Path, accent_codes: Iterable[str]) -> Iterable[tuple[str, dict]]:
    """Yield ``(accent_code, row)`` pairs from downloaded metadata files."""
    for accent_code in accent_codes:
        metadata_path = _metadata_path(data_dir, accent_code)
        with metadata_path.open(encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if line:
                    yield accent_code, json.loads(line)


def _resolve_audio_path(data_dir: Path, accent_code: str, file_name: str) -> Path:
    """Resolve metadata ``file_name`` to a downloaded audio path."""
    direct_path = data_dir / file_name
    if direct_path.exists():
        return direct_path
    old_layout_path = data_dir / accent_code / file_name
    if old_layout_path.exists():
        return old_layout_path
    return data_dir / "test" / accent_code / file_name


def _audio_complete(data_dir: Path, accent_codes: Iterable[str]) -> bool:
    """Check whether every metadata row has a corresponding local audio file."""
    if not _metadata_complete(data_dir, accent_codes):
        return False
    for accent_code, row in _iter_metadata_rows(data_dir, accent_codes):
        if not _resolve_audio_path(data_dir, accent_code, row["file_name"]).exists():
            return False
    return True


def _allow_patterns(accent_codes: Iterable[str], with_audio: bool) -> list[str]:
    """Build HuggingFace snapshot patterns for metadata and optional audio.

    ``score.py`` and ``word_mappings.py`` are fetched alongside the data so the
    official AppTek scorer is available locally for cross-checking the
    NeMo Skills WER against the upstream scorer.
    """
    patterns = ["README.md", "score.py", "word_mappings.py"]
    for accent_code in accent_codes:
        patterns.append(f"test/{accent_code}/metadata.jsonl")
        if with_audio:
            patterns.append(f"test/{accent_code}/audio/*.wav")
    return patterns


def download_dataset(download_dir: Path, accent_codes: Iterable[str], with_audio: bool) -> Path:
    """Download AppTek metadata and optionally audio files from HuggingFace."""
    from huggingface_hub import snapshot_download

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    if _metadata_complete(download_dir, accent_codes) and (
        not with_audio or _audio_complete(download_dir, accent_codes)
    ):
        print(f"Data already exists at {download_dir}. Skipping download.")
        return download_dir

    print("=" * 70)
    print(f"DOWNLOADING {HF_REPO_ID} from HuggingFace")
    print(f"Destination: {download_dir}")
    if with_audio:
        print("Total repository size: 34.9 GB.")
        print("WARNING: This can take a long time depending on network speed.")
    else:
        print("Downloading metadata only because --no-audio was set.")
    print("=" * 70)

    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(download_dir),
        allow_patterns=_allow_patterns(accent_codes, with_audio),
    )
    return download_dir


def _get_audio_duration(audio_path: Path) -> float | None:
    """Read audio duration from a local file if it exists."""
    if not audio_path.exists():
        return None

    import soundfile as sf

    return float(sf.info(str(audio_path)).duration)


def build_messages(audio_path: str, duration: float | None) -> list[dict]:
    """Build OpenAI chat messages with audio metadata."""
    audio: dict = {"path": audio_path}
    if duration is not None:
        audio["duration"] = duration
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": PROMPT, "audio": audio},
    ]


def _format_entry(row: dict, accent_code: str, data_dir: Path, audio_prefix: str) -> dict | None:
    """Format one AppTek metadata row as a NeMo Skills JSONL record.

    Returns ``None`` for rows whose reference transcription is empty.
    """
    expected_answer = row["text"].strip()
    if not expected_answer:
        return None

    file_name = row["file_name"]
    local_audio_path = _resolve_audio_path(data_dir, accent_code, file_name)
    duration = _get_audio_duration(local_audio_path)
    relative_audio_path = local_audio_path.relative_to(data_dir).as_posix()
    audio_path = f"{audio_prefix.rstrip('/')}/{relative_audio_path}"

    entry = {
        "task_type": "ASR",
        "expected_answer": expected_answer,
        "messages": build_messages(audio_path, duration),
        "subset_for_metrics": accent_code,
        "audio_filepath": audio_path,
        "extra_fields": {
            "file_name": file_name,
            "accent_code": accent_code,
            "accent": row["accent"],
            "domain": row["domain"],
            "gender": row["gender"],
        },
    }
    if duration is not None:
        entry["duration"] = duration
    return entry


def _resolve_data_dir(arg_data_dir: str | None) -> Path:
    """Resolve where AppTek raw data lives, following NeMo Skills conventions."""
    if arg_data_dir:
        return Path(arg_data_dir).expanduser().resolve()

    env_dir = os.getenv("NEMO_SKILLS_DATA_DIR")
    if env_dir:
        return (Path(env_dir).expanduser() / BENCHMARK_NAME).resolve()

    pkg_dir = Path(__file__).resolve().parent
    pkg_dir_str = str(pkg_dir)
    if "site-packages" in pkg_dir_str or "dist-packages" in pkg_dir_str:
        raise SystemExit(
            "Missing --data_dir and NEMO_SKILLS_DATA_DIR is not set. "
            "Refusing to write into the installed package directory; please set "
            "NEMO_SKILLS_DATA_DIR or pass --data_dir."
        )
    return pkg_dir


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Prepare AppTek Call-Center Dialogues")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=(
            "Path to the AppTek dataset root. Defaults to "
            "$NEMO_SKILLS_DATA_DIR/apptek-callcenter-dialogues or this package directory."
        ),
    )
    parser.add_argument(
        "--audio-prefix",
        type=str,
        default=DEFAULT_AUDIO_PREFIX,
        help=(
            f"Audio path prefix written into test.jsonl (default: {DEFAULT_AUDIO_PREFIX}, "
            "matches the container mount used on the cluster). Override with an absolute "
            "local path when running outside a container."
        ),
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Download/use metadata only. The generated manifest still contains expected audio paths.",
    )
    parser.add_argument(
        "--accent-codes",
        nargs="+",
        default=list(ACCENT_CODES),
        help="Accent codes to include. Defaults to all accents.",
    )
    return parser.parse_args()


def main() -> None:
    """Download metadata/audio as needed and write the in-repo test split."""
    args = _parse_args()
    unknown_accents = sorted(set(args.accent_codes) - set(ACCENT_CODES))
    if unknown_accents:
        raise ValueError(f"Unknown accent code(s): {', '.join(unknown_accents)}")

    output_dir = Path(__file__).parent
    data_dir = _resolve_data_dir(args.data_dir)
    with_audio = not args.no_audio

    if not _metadata_complete(data_dir, args.accent_codes) or (
        with_audio and not _audio_complete(data_dir, args.accent_codes)
    ):
        download_dataset(data_dir, args.accent_codes, with_audio=with_audio)
    else:
        print(f"Using pre-downloaded data from {data_dir}")

    audio_prefix = args.audio_prefix
    output_file = output_dir / "test.jsonl"

    skipped = 0
    print(f"Writing {output_file}")
    entries = []
    for accent_code, row in tqdm(list(_iter_metadata_rows(data_dir, args.accent_codes)), desc=BENCHMARK_NAME):
        entry = _format_entry(row, accent_code, data_dir, audio_prefix)
        if entry is None:
            skipped += 1
            continue
        entries.append(entry)

    count = len(entries)
    with output_file.open("w", encoding="utf-8") as fout:
        for entry in entries:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(
        f"Prepared {count} samples for {BENCHMARK_NAME}"
        + (f" ({skipped} skipped, empty reference)" if skipped else "")
    )
    print(f"Audio prefix in manifest: {audio_prefix}")


if __name__ == "__main__":
    main()
