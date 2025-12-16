# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

"""Prepare LibriSpeech-PC for ASR evaluation with punctuation and capitalization.

LibriSpeech-PC provides manifests with punctuation/capitalization from OpenSLR-145.
Audio files are downloaded from original LibriSpeech at OpenSLR-12.

Usage:
    ns prepare_data librispeech-pc --data_dir <path_to_data_dir>
    ns prepare_data librispeech-pc --split test-clean (or test-other) --data_dir <path_to_data_dir>
"""

import argparse
import json
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

from tqdm import tqdm


def download_with_progress(url: str, output_path: Path, desc: str):
    """Download file with tqdm progress bar."""
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc=desc) as pbar:

        def reporthook(blocknum, blocksize, totalsize):
            if pbar.total != totalsize:
                pbar.total = totalsize
            downloaded = blocknum * blocksize
            pbar.update(max(0, downloaded - pbar.n))

        urllib.request.urlretrieve(url, output_path, reporthook)


# LibriSpeech-PC manifests (with punctuation and capitalization)
MANIFESTS_URL = "https://www.openslr.org/resources/145/manifests.tar.gz"

# Original LibriSpeech audio files
AUDIO_URLS = {
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "test-other": "https://www.openslr.org/resources/12/test-other.tar.gz",
}


def download_manifests(output_dir: Path) -> Path:
    """Download LibriSpeech-PC manifests if not already present."""
    if (output_dir / "test-clean.json").exists() and (output_dir / "test-other.json").exists():
        return output_dir

    tar_path = output_dir / "manifests.tar.gz"
    download_with_progress(MANIFESTS_URL, tar_path, "Downloading manifests")

    with tarfile.open(tar_path, "r:gz") as tar:
        wanted = {"test-clean.json", "test-other.json"}
        for member in tar.getmembers():
            name = Path(member.name).name
            if name not in wanted:
                continue
            fobj = tar.extractfile(member)
            if fobj is None:
                continue
            out_path = output_dir / name
            with open(out_path, "wb") as fout:
                shutil.copyfileobj(fobj, fout)
    os.remove(tar_path)

    print("✓ Manifests ready\n")
    return output_dir


def download_audio(split: str, audio_dir: Path):
    """Download LibriSpeech audio files if not already present."""
    split_dir = audio_dir / "LibriSpeech" / split.replace("-", "_")
    if split_dir.exists():
        return

    tar_path = audio_dir / f"{split}.tar.gz"
    download_with_progress(AUDIO_URLS[split], tar_path, f"Downloading {split}")

    with tarfile.open(tar_path, "r:gz") as tar:
        if sys.version_info >= (3, 11, 4):
            tar.extractall(audio_dir, filter="data")
        else:
            tar.extractall(audio_dir)
    os.remove(tar_path)


def process_split(split: str, data_dir: Path, audio_dir: Path, with_audio: bool) -> int:
    """Process one LibriSpeech-PC split into nemo-skills format."""

    output_file = data_dir / f"{split}.jsonl"
    manifest_file = data_dir / f"{split}.json"
    if not manifest_file.exists():
        print(f"✗ Manifest not found: {manifest_file}")
        return 0

    if with_audio:
        download_audio(split, audio_dir)

    with open(manifest_file, "r") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    processed = 0
    skipped = 0

    with open(output_file, "w") as fout:
        for entry in entries:
            audio_filepath = entry.get("audio_filepath", "")
            text = entry.get("text", "")

            if not audio_filepath or not text:
                skipped += 1
                continue

            audio_id = Path(audio_filepath).stem

            audio_root = os.getenv("NEMO_SKILLS_AUDIO_ROOT", "/data")
            rel_audio_path = audio_filepath.lstrip("/")
            if rel_audio_path.startswith("LibriSpeech/"):
                rel_audio_path = rel_audio_path[len("LibriSpeech/") :]
            container_path = f"{audio_root}/librispeech-pc/LibriSpeech/{rel_audio_path}"

            user_message = {
                "role": "user",
                "content": "Transcribe the audio with proper punctuation and capitalization.",
                "audio": {"path": container_path},
            }

            output_entry = {
                "audio_filepath": container_path,
                "text": text,
                "expected_answer": text,
                "task_type": "ASR-PC",
                "sample_id": audio_id,
                "split": split,
                "messages": [{"role": "system", "content": "You are a helpful assistant. /no_think"}, user_message],
            }

            fout.write(json.dumps(output_entry, ensure_ascii=False) + "\n")
            processed += 1

    print(f"✓ {split}: {processed} samples" + (f" ({skipped} skipped)" if skipped > 0 else ""))

    if processed > 0 and manifest_file.exists():
        os.remove(manifest_file)

    return processed


def main():
    parser = argparse.ArgumentParser(description="Prepare LibriSpeech-PC for ASR evaluation")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.getenv("NEMO_SKILLS_DATA_DIR"),
        help=(
            "Base data dir (defaults to $NEMO_SKILLS_DATA_DIR). "
            "If provided, output goes under <data_dir>/librispeech-pc. "
            "If omitted, writes into this package's dataset directory (only allowed outside site-packages)."
        ),
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "test-clean", "test-other"],
        help="Which split to prepare (default: all)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip audio download",
    )
    args = parser.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir) / "librispeech-pc"
    else:
        pkg_dir = Path(__file__).parent
        pkg_dir_str = str(pkg_dir)
        if "site-packages" in pkg_dir_str or "dist-packages" in pkg_dir_str:
            raise SystemExit(
                "Missing --data_dir and NEMO_SKILLS_DATA_DIR is not set. "
                "Refusing to write into the installed package directory; please set NEMO_SKILLS_DATA_DIR "
                "or pass --data_dir."
            )
        data_dir = pkg_dir

    audio_dir = data_dir
    audio_dir.mkdir(parents=True, exist_ok=True)

    download_manifests(data_dir)

    splits = ["test-clean", "test-other"] if args.split == "all" else [args.split]
    total = sum(process_split(split, data_dir, audio_dir, not args.no_audio) for split in splits)

    print(f"\n✓ Complete: {total} samples")


if __name__ == "__main__":
    main()
