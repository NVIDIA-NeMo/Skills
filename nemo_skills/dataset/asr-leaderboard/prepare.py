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

"""Prepare ASR Leaderboard datasets for evaluation.

Downloads and formats datasets from the official HF Open ASR Leaderboard ESB
test-only sorted dataset (hf-audio/esb-datasets-test-only-sorted). This is the
same data source used by the official leaderboard, ensuring apples-to-apples
WER comparison.

Audio paths in JSONL: /dataset/asr-leaderboard/data/{dataset}/{sample_id}.flac

Usage:
    ns prepare_data asr-leaderboard
    ns prepare_data asr-leaderboard --datasets librispeech_clean ami
    ns prepare_data asr-leaderboard --no-audio
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from tqdm import tqdm

HF_REPO = "hf-audio/esb-datasets-test-only-sorted"
SYSTEM_MESSAGE = "You are a helpful assistant. /no_think"
AUDIO_SAMPLE_RATE = 16000

# (config, split, text_field, id_field)
DATASET_CONFIGS = {
    "librispeech_clean": ("librispeech", "test.clean", "text", "id"),
    "librispeech_other": ("librispeech", "test.other", "text", "id"),
    "voxpopuli": ("voxpopuli", "test", "text", "id"),
    "tedlium": ("tedlium", "test", "text", "id"),
    "gigaspeech": ("gigaspeech", "test", "text", "id"),
    "spgispeech": ("spgispeech", "test", "text", "id"),
    "earnings22": ("earnings22", "test", "text", "id"),
    "ami": ("ami", "test", "text", "id"),
}


def extract_audio(audio_info):
    """Extract audio array and sampling rate from a HF dataset audio entry.

    Handles both the legacy dict format ({"array": ..., "sampling_rate": ...})
    and the newer AudioDecoder object from torchcodec-based datasets library.
    """
    if audio_info is None:
        return None, None
    try:
        audio_array = np.array(audio_info["array"])
        sampling_rate = int(audio_info["sampling_rate"])
        return audio_array, sampling_rate
    except (KeyError, TypeError, IndexError):
        return None, None


def format_entry(entry, dataset_name, audio_dir, text_field, id_field, with_audio):
    """Format a dataset entry into JSONL and optionally save the audio file."""
    text = entry[text_field].strip()
    if not text:
        return None

    sample_id = str(entry[id_field]).replace("/", "_")
    audio_filename = f"{Path(sample_id).stem}.flac"

    audio_array, sampling_rate = extract_audio(entry.get("audio"))
    duration = None

    if audio_array is not None and sampling_rate is not None:
        duration = len(audio_array) / sampling_rate
        if with_audio:
            sf.write(str(audio_dir / audio_filename), audio_array, sampling_rate)

    user_message = {"role": "user", "content": "Transcribe the following audio."}
    audio_meta = {"path": f"/dataset/asr-leaderboard/data/{dataset_name}/{audio_filename}"}
    if duration is not None:
        audio_meta["duration"] = float(duration)
    user_message["audio"] = audio_meta

    formatted = {
        "task_type": "ASR",
        "expected_answer": text,
        "messages": [{"role": "system", "content": SYSTEM_MESSAGE}, user_message],
        "subset_for_metrics": dataset_name,
        "id": entry[id_field],
    }
    if "speaker_id" in entry:
        formatted["speaker_id"] = entry["speaker_id"]

    return formatted


def prepare_dataset(dataset_name, output_dir, with_audio=True):
    """Download, decode, and write a single ASR dataset to JSONL + audio files."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")

    hf_config, hf_split, text_field, id_field = DATASET_CONFIGS[dataset_name]

    print(f"Loading {dataset_name} from {HF_REPO} (config={hf_config}, split={hf_split})...")
    dataset = load_dataset(HF_REPO, hf_config, split=hf_split)
    if with_audio and "audio" in dataset.column_names:
        dataset = dataset.cast_column("audio", Audio(sampling_rate=AUDIO_SAMPLE_RATE))

    output_file = output_dir / f"{dataset_name}.jsonl"
    audio_dir = output_dir / "data" / dataset_name

    if with_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(dataset)} samples from {dataset_name}...")
    count = 0
    with open(output_file, "w", encoding="utf-8") as fout:
        for entry in tqdm(dataset, desc=dataset_name):
            formatted = format_entry(entry, dataset_name, audio_dir, text_field, id_field, with_audio)
            if formatted is None:
                continue
            fout.write(json.dumps(formatted) + "\n")
            count += 1

    print(f"Saved {count} samples to {output_file}")
    return count


def main():
    parser = argparse.ArgumentParser(description="Prepare ASR Leaderboard datasets for evaluation")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        choices=list(DATASET_CONFIGS.keys()) + ["all"],
        help="Datasets to prepare (default: all)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip saving audio files (JSONL still includes audio paths)",
    )
    args = parser.parse_args()

    data_dir = Path("/dataset/asr-leaderboard")
    output_dir = data_dir if data_dir.exists() else Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with_audio = not args.no_audio
    if not with_audio:
        print("Running without saving audio files.")

    datasets_to_prepare = list(DATASET_CONFIGS.keys()) if "all" in args.datasets else args.datasets

    total_samples = 0
    for dataset_name in datasets_to_prepare:
        total_samples += prepare_dataset(dataset_name, output_dir, with_audio=with_audio)

    combined_file = output_dir / "test.jsonl"
    print(f"\nCreating combined file: {combined_file}")

    dataset_files = sorted(f for f in output_dir.glob("*.jsonl") if f.name != "test.jsonl")
    combined_count = 0
    with open(combined_file, "w", encoding="utf-8") as fout:
        for dataset_file in dataset_files:
            with open(dataset_file, encoding="utf-8") as fin:
                for line in fin:
                    fout.write(line)
                    combined_count += 1
            print(f"  Added {dataset_file.name}")

    print(f"Combined {combined_count} samples from {len(dataset_files)} datasets into {combined_file}")
    print(f"\nTotal: {total_samples} samples prepared")


if __name__ == "__main__":
    main()
