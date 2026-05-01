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

"""Prepare ContextASR-Bench dataset for NeMo Skills evaluation.

Downloads the ContextASR-Speech English subset from HuggingFace (JSONL + 8
audio tar files, ~22 GB total), extracts audio, and produces three JSONL
split files — one per evaluation mode (contextless, coarse, fine).

If --data_dir is provided and already contains the dataset, the download
step is skipped and the existing data is used directly.

Dataset: https://huggingface.co/datasets/MrSupW/ContextASR-Bench

Usage:
    # Auto-download to default location (alongside this script)
    ns prepare_data contextasr-bench

    # Use pre-downloaded data
    ns prepare_data contextasr-bench --data_dir=/path/to/ContextASR-Bench

    # Skip audio download (JSONL only, not recommended)
    ns prepare_data contextasr-bench --no-audio
"""

import argparse
import json
import tarfile
from pathlib import Path

HF_REPO_ID = "MrSupW/ContextASR-Bench"
JSONL_FILENAME = "ContextASR-Speech_English.jsonl"
AUDIO_TAR_PREFIX = "audio/ContextASR-Speech/English/ContextASR-Speech_English"
NUM_AUDIO_TARS = 8

PROMPT_CONTEXTLESS = "Transcribe the English audio into text, ensuring all punctuation marks are included."
PROMPT_COARSE = (
    "This audio belongs to the {domain_label} field. "
    "Transcribe the English audio into text, ensuring all punctuation marks are included."
)
PROMPT_FINE = (
    "This audio belongs to the {domain_label} field and may contain the following "
    "words or phrases: {entity_list}. "
    "Transcribe the English audio into text, ensuring all punctuation marks are included."
)


def download_dataset(download_dir):
    """Download ContextASR-Speech English data from HuggingFace.

    Downloads the JSONL metadata and 8 audio tar files (~22 GB total),
    then extracts the tars. This can take 30-60 minutes depending on
    network speed.
    """
    from huggingface_hub import hf_hub_download

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = download_dir / JSONL_FILENAME
    audio_dir = download_dir / "audio" / "ContextASR-Speech" / "English"

    # Check if data already exists
    if jsonl_path.exists() and audio_dir.exists():
        wav_count = len(list(audio_dir.glob("*.wav")))
        if wav_count > 15000:
            print(f"Dataset already exists at {download_dir} ({wav_count} audio files found). Skipping download.")
            return download_dir

    print("=" * 70)
    print("DOWNLOADING ContextASR-Bench English Speech data from HuggingFace")
    print(f"Repository: {HF_REPO_ID}")
    print(f"Destination: {download_dir}")
    print("Total download size: ~22 GB (JSONL + 8 audio tar files)")
    print("")
    print("WARNING: This download may take 30-60 minutes depending on your")
    print("network speed. You can skip this by pre-downloading the data and")
    print("passing --data_dir=/path/to/ContextASR-Bench.")
    print("=" * 70)

    # Download JSONL
    print(f"\n[1/{NUM_AUDIO_TARS + 1}] Downloading {JSONL_FILENAME}...")
    hf_hub_download(
        repo_id=HF_REPO_ID,
        filename=JSONL_FILENAME,
        repo_type="dataset",
        local_dir=str(download_dir),
    )
    print(f"  Saved to {jsonl_path}")

    # Download and extract audio tars
    audio_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, NUM_AUDIO_TARS + 1):
        tar_filename = f"{AUDIO_TAR_PREFIX}_{i}.tar"
        print(f"\n[{i + 1}/{NUM_AUDIO_TARS + 1}] Downloading {tar_filename}...")

        local_tar = Path(
            hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=tar_filename,
                repo_type="dataset",
                local_dir=str(download_dir),
            )
        )

        print(f"  Extracting {local_tar.name} to {audio_dir}...")
        with tarfile.open(local_tar) as tf:
            tf.extractall(path=audio_dir, filter="data")

        print("  Extracted. Removing tar file to save space...")
        local_tar.unlink()

    wav_count = len(list(audio_dir.glob("*.wav")))
    print(f"\nDownload complete! {wav_count} audio files extracted to {audio_dir}")

    return download_dir


def build_messages(prompt_text, audio_path, duration):
    """Build OpenAI-format messages with audio metadata."""
    return [
        {
            "role": "user",
            "content": prompt_text,
            "audio": {
                "path": audio_path,
                "duration": float(duration),
            },
        }
    ]


def format_entry(sample, mode, audio_prefix):
    """Format a single dataset sample into a JSONL record for a given mode."""
    audio_path = f"{audio_prefix}/{sample['audio']}"
    entity_list = sample["entity_list"]
    domain_label = sample["domain_label"]

    if mode == "contextless":
        prompt = PROMPT_CONTEXTLESS
    elif mode == "coarse":
        prompt = PROMPT_COARSE.format(domain_label=domain_label)
    elif mode == "fine":
        entity_str = ", ".join(entity_list)
        prompt = PROMPT_FINE.format(domain_label=domain_label, entity_list=entity_str)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {
        "messages": build_messages(prompt, audio_path, sample["duration"]),
        "expected_answer": sample["text"],
        "entity_list": entity_list,
        "domain_label": domain_label,
        "subset_for_metrics": domain_label,
        "uniq_id": sample["uniq_id"],
        "duration": float(sample["duration"]),
        "audio_filepath": audio_path,
    }


def main():
    """Parse arguments, download data if needed, and write per-mode JSONL splits."""
    parser = argparse.ArgumentParser(description="Prepare ContextASR-Bench for NeMo Skills")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=(
            "Path to ContextASR-Bench dataset root. If the directory already contains "
            "ContextASR-Speech_English.jsonl, that data is used directly. If the "
            "directory is empty or the file is missing, data will be downloaded there "
            "automatically from HuggingFace. "
            "If not provided, downloads to a 'data/' subdirectory next to this script."
        ),
    )
    parser.add_argument(
        "--audio-prefix",
        type=str,
        default=None,
        help=(
            "Override audio path prefix written into JSONL files. "
            "Defaults to the data_dir value. Useful for container mount points "
            "(e.g., --audio-prefix /data/contextasr-bench)."
        ),
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip downloading/verifying audio files (not recommended for actual evaluation)",
    )
    args = parser.parse_args()

    output_dir = Path(__file__).parent

    data_dir = Path(args.data_dir) if args.data_dir else output_dir / "data"
    jsonl_path = data_dir / JSONL_FILENAME

    if jsonl_path.exists():
        print(f"Using pre-downloaded data from {data_dir}")
    elif args.no_audio:
        raise FileNotFoundError(
            f"Dataset file not found: {jsonl_path}\n"
            f"Cannot use --no-audio when data has not been downloaded yet. "
            f"Either run without --no-audio first to download, or point --data_dir "
            f"to a directory that already contains {JSONL_FILENAME}."
        )
    else:
        print(f"Data not found at {data_dir}. Downloading there...")
        download_dataset(data_dir)

    audio_prefix = args.audio_prefix if args.audio_prefix else str(data_dir)
    audio_prefix = audio_prefix.rstrip("/")

    jsonl_path = data_dir / JSONL_FILENAME

    print(f"\nReading dataset from {jsonl_path}")
    samples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    print(f"Loaded {len(samples)} samples")

    if not args.no_audio:
        sample_audio = Path(audio_prefix) / samples[0]["audio"]
        if not sample_audio.exists():
            print(
                f"WARNING: Sample audio file not found at {sample_audio}. "
                f"Audio paths may need adjustment via --audio-prefix."
            )
        else:
            print(f"Audio files verified (sample check: {sample_audio})")

    modes = {
        "contextless": output_dir / "contextless" / "test.jsonl",
        "coarse": output_dir / "coarse" / "test.jsonl",
        "fine": output_dir / "fine" / "test.jsonl",
    }

    print("\nWriting JSONL splits...")
    for mode_name, output_path in modes.items():
        entries = [format_entry(sample, mode_name, audio_prefix) for sample in samples]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fout:
            for entry in entries:
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"  {mode_name}: wrote {len(entries)} samples to {output_path}")

    print(f"\nDone. Total: {len(samples)} samples x 3 modes = {len(samples) * 3} records")


if __name__ == "__main__":
    main()
