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
same data source used by the official leaderboard and the offline NeMo eval
pipeline, ensuring apples-to-apples WER comparison.

Audio paths in JSONL: /dataset/asr-leaderboard/data/{dataset}/{sample_id}.flac

Usage:
    ns prepare_data asr-leaderboard
    ns prepare_data asr-leaderboard --datasets librispeech_clean ami
    ns prepare_data asr-leaderboard --datasets earnings22
    ns prepare_data asr-leaderboard --no-audio  # skip saving audio files
"""

import argparse
import json
from pathlib import Path

import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

SYSTEM_MESSAGE = "You are a helpful assistant. /no_think"
MIN_AUDIO_DURATION = 0.1  # Skip audio shorter than this (causes mel spectrogram errors)

# (hf_repo, config, split, text_field, id_field)
DATASET_CONFIGS = {
    "librispeech_clean": ("hf-audio/esb-datasets-test-only-sorted", "librispeech", "test.clean", "text", "id"),
    "librispeech_other": ("hf-audio/esb-datasets-test-only-sorted", "librispeech", "test.other", "text", "id"),
    "voxpopuli": ("hf-audio/esb-datasets-test-only-sorted", "voxpopuli", "test", "text", "id"),
    "tedlium": ("hf-audio/esb-datasets-test-only-sorted", "tedlium", "test", "text", "id"),
    "gigaspeech": ("hf-audio/esb-datasets-test-only-sorted", "gigaspeech", "test", "text", "id"),
    "spgispeech": ("hf-audio/esb-datasets-test-only-sorted", "spgispeech", "test", "text", "id"),
    "earnings22": ("hf-audio/esb-datasets-test-only-sorted", "earnings22", "test", "text", "id"),
    "ami": ("hf-audio/esb-datasets-test-only-sorted", "ami", "test", "text", "id"),
}


def save_audio_and_format_entry(
    entry, dataset_name, audio_dir, sample_idx, text_field="text", id_field="id", with_audio=True
):
    """Format a dataset entry and optionally save audio file."""
    text = entry[text_field].strip()

    system_message = {"role": "system", "content": SYSTEM_MESSAGE}
    user_message = {"role": "user", "content": "Transcribe the following audio."}

    sample_id = str(entry[id_field]).replace("/", "_")
    audio_filename = f"{Path(sample_id).stem}.flac"

    audio_info = entry.get("audio", {})
    duration = None
    if isinstance(audio_info, dict) and "array" in audio_info and "sampling_rate" in audio_info:
        audio_array = audio_info["array"]
        sampling_rate = audio_info["sampling_rate"]
        duration = len(audio_array) / sampling_rate

        if duration < MIN_AUDIO_DURATION:
            return None

        if with_audio:
            sf.write(str(audio_dir / audio_filename), audio_array, sampling_rate)

    audio_meta = {"path": f"/dataset/asr-leaderboard/data/{dataset_name}/{audio_filename}"}
    if duration is not None:
        audio_meta["duration"] = float(duration)
    user_message["audio"] = audio_meta

    formatted_entry = {
        "task_type": "ASR",
        "expected_answer": text,
        "messages": [system_message, user_message],
        "subset_for_metrics": dataset_name,
    }

    formatted_entry["id"] = entry[id_field]
    if "speaker_id" in entry:
        formatted_entry["speaker_id"] = entry["speaker_id"]

    return formatted_entry


def prepare_dataset(dataset_name, output_dir, with_audio=True):
    """Prepare a single ASR dataset."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")

    hf_repo, hf_config, hf_split, text_field, id_field = DATASET_CONFIGS[dataset_name]

    print(f"Loading {dataset_name} from {hf_repo} (config={hf_config}, split={hf_split})...")
    dataset = load_dataset(hf_repo, hf_config, split=hf_split, trust_remote_code=True)

    output_file = output_dir / f"{dataset_name}.jsonl"
    audio_dir = output_dir / "data" / dataset_name

    if with_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving audio files to {audio_dir}")

    print(f"Processing {len(dataset)} samples from {dataset_name}...")

    count = 0
    skipped = 0
    with open(output_file, "w", encoding="utf-8") as fout:
        for idx, entry in enumerate(tqdm(dataset, desc=dataset_name)):
            formatted = save_audio_and_format_entry(
                entry, dataset_name, audio_dir, idx, text_field=text_field, id_field=id_field, with_audio=with_audio
            )
            if formatted is None:
                skipped += 1
                continue
            if formatted["expected_answer"]:
                fout.write(json.dumps(formatted) + "\n")
                count += 1

    if skipped > 0:
        print(f"Skipped {skipped} samples with audio < {MIN_AUDIO_DURATION}s")

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

    if args.no_audio:
        print("Running without saving audio files.")
    else:
        print("Running with audio. Saving to data/{dataset}/")

    datasets_to_prepare = list(DATASET_CONFIGS.keys()) if "all" in args.datasets else args.datasets

    total_samples = 0
    for dataset_name in datasets_to_prepare:
        total_samples += prepare_dataset(dataset_name, output_dir, with_audio=with_audio)

    # Combine all dataset JSONLs into test.jsonl
    combined_file = output_dir / "test.jsonl"
    print(f"\nCreating combined file: {combined_file}")

    all_jsonl_files = sorted(output_dir.glob("*.jsonl"))
    dataset_files = [f for f in all_jsonl_files if f.name != "test.jsonl"]

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
