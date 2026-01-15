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

Downloads and formats datasets from the HuggingFace Open ASR Leaderboard.
Audio paths in JSONL: /dataset/asr-leaderboard/data/{dataset}/{sample_id}.flac

Usage:
    ns prepare_data asr-leaderboard
    ns prepare_data asr-leaderboard --datasets librispeech_clean ami
    ns prepare_data asr-leaderboard --no-audio  # skip saving audio files
"""

import argparse
import json
from pathlib import Path

import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

SYSTEM_MESSAGE = "You are a helpful assistant. /no_think"
USER_MESSAGE = "Transcribe the audio file into English text."
MIN_AUDIO_DURATION = 0.1  # Skip audio shorter than this

# Speaker IDs to skip in Tedlium dataset
SKIP_SPEAKER_IDS = {"inter_segment_gap"}

# Non-speech tokens to skip in GigaSpeech dataset
NONSPEECH_TOKENS = {"<SIL>", "<MUSIC>", "<NOISE>", "<OTHER>"}


def is_nonspeech_only(text):
    """Check if text contains only non-speech tokens."""
    tokens = set(text.strip().split())
    return tokens and tokens.issubset(NONSPEECH_TOKENS)


# (hf_dataset, hf_config, hf_split, streaming)
DATASET_CONFIGS = {
    "librispeech_clean": ("librispeech_asr", "clean", "test", False),
    "librispeech_other": ("librispeech_asr", "other", "test", False),
    "voxpopuli": ("facebook/voxpopuli", "en", "test", False),
    "tedlium": ("LIUM/tedlium", "release3", "test", False),
    "gigaspeech": ("speechcolab/gigaspeech", "xs", "test", False),
    "spgispeech": ("kensho/spgispeech", "test", "test", True),  # streaming to avoid timeout due to large metadata
    "earnings22": ("distil-whisper/earnings22", "chunked", "test", False),
    "ami": ("edinburghcstr/ami", "ihm", "test", False),
}


def save_audio_and_format_entry(entry, dataset_name, audio_dir, sample_idx, with_audio=True):
    """Format a dataset entry and optionally save audio file."""
    # Different datasets use different field names for transcription
    text = (
        entry.get("text", "")  # ami, LS, gigaspeech, tedlium
        or entry.get("normalized_text", "")  # voxpopuli
        or entry.get("transcript", "")  # spgispeech
        or entry.get("transcription", "")  # earnings22
    )
    text = text.strip() if text else ""

    system_message = {"role": "system", "content": SYSTEM_MESSAGE}
    user_message = {"role": "user", "content": USER_MESSAGE}

    audio_info = entry.get("audio", {})
    if isinstance(audio_info, dict) and "array" in audio_info and "sampling_rate" in audio_info:
        audio_array = audio_info["array"]
        sampling_rate = audio_info["sampling_rate"]

        # Skip if audio array is empty or invalid
        if audio_array is None or len(audio_array) == 0:
            return None

        duration = len(audio_array) / sampling_rate

        if duration < MIN_AUDIO_DURATION:
            return None

        sample_id = entry.get("id", str(sample_idx))
        audio_filename = f"{sample_id}.flac"

        if with_audio:
            sf.write(str(audio_dir / audio_filename), audio_array, sampling_rate)

        audio_filepath = f"/dataset/asr-leaderboard/data/{dataset_name}/{audio_filename}"
        user_message["audio"] = {
            "path": audio_filepath,
            "duration": float(duration),
        }

    formatted_entry = {
        "task_type": "ASR",
        "expected_answer": text,
        "messages": [system_message, user_message],
        "subset_for_metrics": dataset_name,
    }

    # Add audio_filepath and duration as top-level fields
    if "audio" in user_message:
        formatted_entry["audio_filepath"] = user_message["audio"]["path"]
        formatted_entry["duration"] = user_message["audio"]["duration"]

    if "id" in entry:
        formatted_entry["id"] = entry["id"]
    if "speaker_id" in entry:
        formatted_entry["speaker_id"] = entry["speaker_id"]

    return formatted_entry


def prepare_dataset(dataset_name, output_dir, with_audio=True):
    """Prepare a single ASR dataset."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")

    hf_dataset, hf_config, hf_split, streaming = DATASET_CONFIGS[dataset_name]

    print(f"Loading {dataset_name} from {hf_dataset} (streaming={streaming})...")
    try:
        if hf_config:
            dataset = load_dataset(hf_dataset, hf_config, split=hf_split, trust_remote_code=True, streaming=streaming)
        else:
            dataset = load_dataset(hf_dataset, split=hf_split, trust_remote_code=True, streaming=streaming)
    except Exception as e:
        print(f"Warning: Failed to load {dataset_name}: {e}")
        return 0

    output_file = output_dir / f"{dataset_name}.jsonl"
    audio_dir = output_dir / "data" / dataset_name

    if with_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving audio files to {audio_dir}")

    if streaming:
        print(f"Processing {dataset_name} (streaming)...")
    else:
        print(f"Processing {len(dataset)} samples from {dataset_name}...")

    count = 0
    skipped = 0
    with open(output_file, "w", encoding="utf-8") as fout:
        for idx, entry in enumerate(tqdm(dataset, desc=dataset_name)):
            formatted = save_audio_and_format_entry(entry, dataset_name, audio_dir, idx, with_audio=with_audio)
            if formatted is None:
                skipped += 1
                continue
            # Skip empty answers, non-speech segments, and non-speech-only samples
            speaker_id = entry.get("speaker_id", "")
            expected = formatted["expected_answer"]
            if expected and speaker_id not in SKIP_SPEAKER_IDS and not is_nonspeech_only(expected):
                fout.write(json.dumps(formatted) + "\n")
                count += 1
            else:
                skipped += 1

    if skipped > 0:
        print(f"Skipped {skipped} samples (short audio, non-speech, or invalid)")

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

    output_dir = Path(__file__).parent
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
