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

"""MUSAN Dataset Preparation for nemo-skills

Prepares the MUSAN dataset (Music, Speech, and Noise Corpus) for use with nemo-skills.

Dataset sources:
  - HuggingFace: 774 samples (~5h), incomplete, fast download
  - Kaggle: 2016 files (10.3GB), nearly complete, requires API key
  - OpenSLR: Complete dataset (11GB), official source

Usage:
    python -m nemo_skills.dataset.musan.prepare --source kaggle --categories noise
    python -m nemo_skills.dataset.musan.prepare --categories noise --max-samples 100
    python -m nemo_skills.dataset.musan.prepare --source openslr --categories noise
"""

import argparse
import json
import os
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
from tqdm import tqdm

# HuggingFace dataset label mappings
CATEGORY_LABELS = {
    "noise": 0,
    "music": 1,
}

LABEL_TO_CATEGORY = {
    0: "noise",
    1: "other",
}


def download_from_kaggle(output_dir: Path) -> Path:
    """Download MUSAN dataset from Kaggle using kagglehub."""
    try:
        import kagglehub
    except ImportError:
        raise ImportError("kagglehub not installed. Run: pip install kagglehub")

    print("Downloading from Kaggle (requires API key in ~/.kaggle/kaggle.json)")

    try:
        path = kagglehub.dataset_download("dogrose/musan-dataset")
        print(f"Downloaded to: {path}")
        return Path(path)
    except Exception as e:
        raise Exception(f"Kaggle download failed: {e}")


def download_from_openslr(output_dir: Path) -> Path:
    """Download MUSAN dataset from OpenSLR (11 GB)."""
    url = "https://www.openslr.org/resources/17/musan.tar.gz"
    download_path = output_dir / "musan.tar.gz"
    extract_path = output_dir / "musan_openslr"

    print("Downloading from OpenSLR (~11 GB)")
    print(f"URL: {url}")

    if not download_path.exists():

        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded / total_size * 100, 100)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r{percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")

        urllib.request.urlretrieve(url, download_path, reporthook)
        print("\nDownload complete")
    else:
        print(f"Using cached archive: {download_path}")

    if not extract_path.exists():
        print(f"Extracting to {extract_path}...")
        extract_path.mkdir(parents=True, exist_ok=True)
        with tarfile.open(download_path, "r:gz") as tar:
            if sys.version_info >= (3, 11, 4):
                tar.extractall(extract_path, filter="data")
            else:
                tar.extractall(extract_path)
        print("Extraction complete")
    else:
        print(f"Using extracted data: {extract_path}")

    return extract_path / "musan"


def load_dataset_from_source(source: str, output_dir: Path):
    """Load MUSAN dataset from specified source."""
    if source == "huggingface":
        from datasets import load_dataset

        print("Loading from HuggingFace...")
        dataset = load_dataset("FluidInference/musan", split="train")
        print(f"Loaded {len(dataset)} samples")
        return dataset, "huggingface"

    elif source == "kaggle":
        dataset_path = download_from_kaggle(output_dir)
        musan_path = dataset_path / "musan"
        if not musan_path.exists():
            raise ValueError(f"'musan' directory not found in {dataset_path}")

        print(f"Dataset path: {musan_path}")
        for cat in ["music", "speech", "noise"]:
            cat_path = musan_path / cat
            if cat_path.exists():
                wav_count = len(list(cat_path.glob("**/*.wav")))
                print(f"  {cat}: {wav_count} files")

        return musan_path, "kaggle"

    elif source == "openslr":
        dataset_path = download_from_openslr(output_dir)
        print(f"Dataset path: {dataset_path}")
        for cat in ["music", "speech", "noise"]:
            cat_path = dataset_path / cat
            if cat_path.exists():
                wav_count = len(list(cat_path.glob("**/*.wav")))
                print(f"  {cat}: {wav_count} files")

        return dataset_path, "openslr"

    else:
        raise ValueError(f"Unknown source: {source}")


def get_audio_duration(audio_array: np.ndarray, sampling_rate: int) -> float:
    """Compute audio duration in seconds."""
    if audio_array is None or len(audio_array) == 0:
        return 0.0
    return float(len(audio_array) / sampling_rate)


def save_audio_file(audio_array: np.ndarray, sampling_rate: int, output_path: str):
    """Save audio array to WAV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio_array, sampling_rate)


def create_manifest_entry(
    audio_filename: str,
    duration: float,
    category: str,
    sample_id: int,
    label: str,
) -> Dict:
    """Create nemo-skills manifest entry."""
    audio_root = os.getenv("NEMO_SKILLS_AUDIO_ROOT", "/data")
    audio_rel_path = f"{audio_root}/musan/{category}/audio/{audio_filename}"
    audio_metadata = {"path": audio_rel_path, "duration": duration}

    # Instruction for transcription (expects empty response for non-speech audio)
    instruction = "Transcribe the speech in this audio. If there is no speech, do not output anything."

    entry = {
        "audio_path": [audio_rel_path],
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. /no_think"},
            {
                "role": "user",
                "content": instruction,
                "audio": audio_metadata,
                "audios": [audio_metadata],
            },
        ],
        "expected_answer": "",
        "dataset": "musan",
        "subset_for_metrics": f"musan_{category}",
        "sample_id": sample_id,
        "category": category,
        "original_label": label,
        "task_type": "Hallucination",
        "audio_duration": duration,
        "question": instruction,
    }

    return entry


def process_category_from_files(
    category: str,
    dataset_path: Path,
    output_dir: Path,
    save_audio: bool = True,
    split: str = "train",
    max_samples: int = -1,
) -> tuple[int, List[Dict]]:
    """Process MUSAN category from WAV files (Kaggle/OpenSLR format)."""
    category_path = dataset_path / category
    if not category_path.exists():
        raise ValueError(f"Category directory not found: {category_path}")

    wav_files = sorted(list(category_path.glob("**/*.wav")))
    print(f"Found {len(wav_files)} WAV files")

    if len(wav_files) == 0:
        return 0, []

    if max_samples > 0 and len(wav_files) > max_samples:
        wav_files = wav_files[:max_samples]
        print(f"Limited to {max_samples} samples")

    audio_dir = output_dir / category / "audio"
    dataset_dir = output_dir / category
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    manifest_entries = []
    successful = 0
    failed = 0

    for idx, wav_path in enumerate(tqdm(wav_files, desc=f"Processing {category}")):
        try:
            audio_array, sampling_rate = sf.read(str(wav_path))
            duration = get_audio_duration(audio_array, sampling_rate)
            audio_filename = f"musan_{category}_{idx:06d}.wav"
            local_audio_path = audio_dir / audio_filename

            if save_audio:
                try:
                    save_audio_file(audio_array, sampling_rate, str(local_audio_path))
                except Exception as e:
                    print(f"Failed to save sample {idx}: {e}")
                    failed += 1
                    continue

            entry = create_manifest_entry(
                audio_filename=audio_filename,
                duration=duration,
                category=category,
                sample_id=idx,
                label=wav_path.stem,
            )

            manifest_entries.append(entry)
            successful += 1

        except Exception as e:
            print(f"Error processing {wav_path}: {e}")
            failed += 1
            continue

    manifest_path = dataset_dir / "test.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved {successful} samples to {manifest_path}")
    if failed > 0:
        print(f"Failed: {failed} samples")

    return successful, manifest_entries


def process_category(
    category: str,
    output_dir: Path,
    dataset,
    source_type: str,
    save_audio: bool = True,
    split: str = "train",
    max_samples: int = -1,
) -> tuple[int, List[Dict]]:
    """Process a single MUSAN category."""
    print(f"\n{'=' * 60}")
    print(f"Processing: {category}")
    print(f"{'=' * 60}")

    if source_type in ["kaggle", "openslr"]:
        return process_category_from_files(
            category=category,
            dataset_path=dataset,
            output_dir=output_dir,
            save_audio=save_audio,
            split=split,
            max_samples=max_samples,
        )

    elif source_type != "huggingface":
        raise NotImplementedError(f"Source '{source_type}' not supported")

    filtered_samples = []
    target_label = CATEGORY_LABELS.get(category)
    if target_label is None:
        print(f"Unknown category '{category}'")
        return 0, []

    for sample in dataset:
        label = sample.get("label")
        if label == target_label:
            filtered_samples.append(sample)

    print(f"Found {len(filtered_samples)} samples")

    if len(filtered_samples) == 0:
        return 0, []

    if max_samples > 0 and len(filtered_samples) > max_samples:
        filtered_samples = filtered_samples[:max_samples]
        print(f"Limited to {max_samples} samples")

    # Create output directories
    audio_dir = output_dir / category / "audio"
    dataset_dir = output_dir / category
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    manifest_entries = []
    successful = 0
    failed = 0

    for idx, sample in enumerate(tqdm(filtered_samples, desc=f"Processing {category}")):
        try:
            audio_dict = sample.get("audio")
            if audio_dict is None:
                failed += 1
                continue

            if isinstance(audio_dict, dict):
                audio_array = audio_dict.get("array")
                sampling_rate = audio_dict.get("sampling_rate", 16000)
            else:
                failed += 1
                continue

            if audio_array is None or len(audio_array) == 0:
                failed += 1
                continue

            if isinstance(audio_array, list):
                audio_array = np.array(audio_array)

            duration = get_audio_duration(audio_array, sampling_rate)
            audio_filename = f"musan_{category}_{idx:06d}.wav"
            local_audio_path = audio_dir / audio_filename

            if save_audio:
                try:
                    save_audio_file(audio_array, sampling_rate, str(local_audio_path))
                except Exception as e:
                    print(f"Failed to save sample {idx}: {e}")
                    failed += 1
                    continue

            label = sample.get("label", -1)
            entry = create_manifest_entry(
                audio_filename=audio_filename,
                duration=duration,
                category=category,
                sample_id=idx,
                label=str(label),
            )

            manifest_entries.append(entry)
            successful += 1

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            failed += 1
            continue

    manifest_path = dataset_dir / "test.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved {successful} samples to {manifest_path}")
    if failed > 0:
        print(f"Failed: {failed} samples")

    return successful, manifest_entries


def main():
    parser = argparse.ArgumentParser(description="Prepare MUSAN dataset for nemo-skills")
    parser.add_argument(
        "--source",
        choices=["huggingface", "kaggle", "openslr"],
        default="huggingface",
        help="Download source: huggingface (fast, incomplete), kaggle (complete, API key), openslr (complete, 11GB)",
    )
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=["music", "speech", "noise"],
        default=["music", "speech", "noise"],
    )
    parser.add_argument("--no-audio", dest="save_audio", action="store_false")
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.set_defaults(save_audio=True)

    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("MUSAN Dataset Preparation")
    print("=" * 60)
    print(f"Source: {args.source}")
    print(f"Output: {output_dir}")
    print(f"Categories: {', '.join(args.categories)}")
    print("=" * 60 + "\n")

    try:
        dataset, source_type = load_dataset_from_source(args.source, output_dir)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    total_samples = 0
    successful_categories = []
    failed_categories = []
    all_entries = []

    for category in args.categories:
        try:
            num_samples, entries = process_category(
                category=category,
                output_dir=output_dir,
                dataset=dataset,
                source_type=source_type,
                save_audio=args.save_audio,
                split=args.split,
                max_samples=args.max_samples,
            )
            total_samples += num_samples
            successful_categories.append(category)
            all_entries.extend(entries)

        except Exception as e:
            print(f"\nFailed: {category} - {e}\n")
            failed_categories.append((category, str(e)))

    if all_entries:
        combined_manifest_path = output_dir / "test.jsonl"
        with open(combined_manifest_path, "w", encoding="utf-8") as f:
            for entry in all_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\nCombined manifest: {combined_manifest_path}")
        print(f"Total samples: {len(all_entries)}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"Requested: {len(args.categories)}, Successful: {len(successful_categories)}, Failed: {len(failed_categories)}"
    )
    print(f"Total samples: {total_samples}")

    if successful_categories:
        for name in successful_categories:
            print(f"  ✓ {name}")

    if failed_categories:
        for name, error in failed_categories:
            print(f"  ✗ {name}: {error}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
