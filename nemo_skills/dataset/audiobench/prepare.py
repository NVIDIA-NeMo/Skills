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

"""AudioBench Dataset Preparation for nemo-skills

This script prepares AudioBench datasets for evaluation with nemo-skills.
AudioBench is a comprehensive benchmark for evaluating speech and audio models
across multiple tasks including ASR, translation, speech QA, and more.

Usage:
    python -m nemo_skills.dataset.audiobench.prepare --split test
    python -m nemo_skills.dataset.audiobench.prepare --datasets librispeech earnings21
    python -m nemo_skills.dataset.audiobench.prepare --category nonjudge
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
from tqdm import tqdm

# AudioBench datasets categorized by evaluation type
JUDGE_DATASETS = [
    "alpaca_audio",
    "audiocaps_qa",
    "audiocaps",
    "clotho_aqa",
    "cn_college_listen_mcq",
    "dream_tts_mcq",
    "iemocap_emotion",
    "iemocap_gender",
    "imda_ar_dialogue",
    "imda_ar_sentence",
    "imda_gr_dialogue",
    "imda_gr_sentence",
    "imda_part3_30s_ds_human",
    "imda_part4_30s_ds_human",
    "imda_part5_30s_ds_human",
    "imda_part6_30s_ds_human",
    "imda_part3_30s_sqa_human",
    "imda_part4_30s_sqa_human",
    "imda_part5_30s_sqa_human",
    "imda_part6_30s_sqa_human",
    "meld_emotion",
    "meld_sentiment",
    "mmau_mini",
    "muchomusic",
    "openhermes_audio",
    "public_sg_speech_qa",
    "slue_p2_sqa5",
    "spoken_squad",
    "voxceleb_accent",
    "voxceleb_gender",
    "wavcaps_qa",
    "wavcaps",
]

NONJUDGE_DATASETS = [
    "aishell_asr_zh",
    "common_voice_15_en",
    "covost2_en_id",
    "covost2_en_ta",
    "covost2_en_zh",
    "covost2_id_en",
    "covost2_ta_en",
    "covost2_zh_en",
    "earnings21",
    "earnings22",
    "gigaspeech",
    "gigaspeech2_indo",
    "gigaspeech2_thai",
    "gigaspeech2_viet",
    "imda_part1_asr",
    "imda_part2_asr",
    "imda_part3_30s_asr",
    "imda_part4_30s_asr",
    "imda_part5_30s_asr",
    "imda_part6_30s_asr",
    "librispeech_test_clean",
    "librispeech_test_other",
    "peoples_speech",
    "seame_dev_man",
    "seame_dev_sge",
    "spoken_mqa_long_digit",
    "spoken_mqa_multi_step_reasoning",
    "spoken_mqa_short_digit",
    "spoken_mqa_single_step_reasoning",
    "tedlium3",
    "tedlium3_long_form",
]


def get_audio_duration(audio_array: np.ndarray, sampling_rate: int) -> float:
    """Compute audio duration in seconds from array and sampling rate."""
    if audio_array is None or len(audio_array) == 0:
        return 0.0
    return float(len(audio_array) / sampling_rate)


def save_audio_file(audio_array: np.ndarray, sampling_rate: int, output_path: str):
    """Save audio array to WAV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio_array, sampling_rate)


def create_manifest_entry(
    sample: Dict,
    audio_filename: str,
    duration: float,
    dataset_name: str,
    sample_id: int,
    category: str,
) -> Dict:
    """Create a nemo-skills compatible manifest entry.
    
    Args:
        sample: Raw sample from AudioBench dataset
        audio_filename: Audio filename (relative path within audiobench directory)
        duration: Audio duration in seconds
        dataset_name: Name of the dataset
        sample_id: Sample index
        category: Category (judge/nonjudge)
    
    Returns:
        Manifest entry dict with proper format for nemo-skills
    """
    instruction = sample.get("instruction", "Process the audio")
    reference = sample.get("reference", "")
    task_type = sample.get("task_type", "unknown")
    
    # Create relative audio path (will be resolved relative to data_dir)
    # Format: audiobench/{category}/audio/{dataset_name}/{filename}
    audio_rel_path = f"audiobench/{category}/audio/{dataset_name}/{audio_filename}"
    
    # Create audio metadata (both singular and plural forms for compatibility)
    audio_metadata = {"path": audio_rel_path, "duration": duration}
    
    entry = {
        "expected_answer": reference,
        "audio_path": [audio_rel_path],
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. /no_think"
            },
            {
                "role": "user",
                "content": instruction,
                "audio": audio_metadata,  # Singular for Megatron server
                "audios": [audio_metadata],  # Plural for compatibility
            }
        ],
        "dataset": dataset_name,
        "subset_for_metrics": dataset_name,
        "sample_id": sample_id,
        "task_type": task_type,
        "question": instruction,
    }
    
    # Add optional fields if present
    for key in ["choices", "options", "audio_text_instruction"]:
        if key in sample:
            entry[key] = sample[key]
    
    return entry


def clone_audiobench_repo(target_dir: Path) -> bool:
    """Clone AudioBench repository if it doesn't exist.
    
    Args:
        target_dir: Directory where AudioBench should be cloned
    
    Returns:
        True if successful, False otherwise
    """
    audiobench_url = "https://github.com/AudioLLMs/AudioBench.git"
    
    if target_dir.exists():
        print(f"AudioBench already exists at {target_dir}")
        return True
    
    print(f"\nCloning AudioBench repository to {target_dir}...")
    print(f"This may take a few minutes...")
    
    try:
        subprocess.run(
            ["git", "clone", audiobench_url, str(target_dir)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"✓ Successfully cloned AudioBench")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to clone AudioBench: {e.stderr}")
        return False
    except FileNotFoundError:
        print("✗ git command not found. Please install git or manually clone AudioBench.")
        return False


def process_dataset(
    dataset_name: str,
    output_dir: Path,
    save_audio: bool = True,
    split: str = "test",
    audiobench_path: Path = None,
    max_samples: int = -1,
) -> tuple[int, List[Dict]]:
    """Process a single AudioBench dataset.
    
    Args:
        dataset_name: Name of the dataset to process
        output_dir: Base output directory
        save_audio: Whether to save audio files
        split: Dataset split (default: "test")
        audiobench_path: Path to AudioBench repository
    
    Returns:
        Tuple of (num_samples, manifest_entries)
    """
    print(f"\n{'='*60}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*60}")
    
    # Import AudioBench Dataset class
    sys.path.insert(0, str(audiobench_path / "src"))
    try:
        from dataset import Dataset
    except ImportError as e:
        raise ImportError(
            f"Failed to import AudioBench Dataset class: {e}\n"
            f"AudioBench path: {audiobench_path}\n"
            f"Make sure AudioBench repository is properly set up."
        )
    
    # Load dataset
    try:
        dataset = Dataset(dataset_name, number_of_samples=max_samples)
        data_samples = dataset.input_data
        print(f"Loaded {len(data_samples)} samples via AudioBench")
    except Exception as e:
        raise Exception(f"Failed to load dataset {dataset_name}: {e}")
    
    # Determine category (handle _test suffix variants)
    dataset_base = dataset_name.replace("_test", "")
    if dataset_name in JUDGE_DATASETS or dataset_base in JUDGE_DATASETS:
        category = "judge"
    elif dataset_name in NONJUDGE_DATASETS or dataset_base in NONJUDGE_DATASETS:
        category = "nonjudge"
    else:
        category = "unknown"
    
    # Create output directories
    audio_dir = output_dir / category / "audio" / dataset_name
    manifest_dir = output_dir / category / "manifests"
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    
    manifest_entries = []
    successful = 0
    failed = 0
    
    for idx, sample in enumerate(tqdm(data_samples, desc=f"Processing {dataset_name}")):
        try:
            # Get audio data
            audio_dict = sample.get("audio")
            if audio_dict is None:
                print(f"Warning: Sample {idx} has no audio, skipping")
                failed += 1
                continue
            
            # Extract audio array and sampling rate
            if isinstance(audio_dict, dict):
                audio_array = audio_dict.get("array")
                sampling_rate = audio_dict.get("sampling_rate", 16000)
            else:
                print(f"Warning: Unexpected audio format at sample {idx}")
                failed += 1
                continue
            
            if audio_array is None or len(audio_array) == 0:
                print(f"Warning: Empty audio at sample {idx}, skipping")
                failed += 1
                continue
            
            # Convert to numpy array if needed
            if isinstance(audio_array, list):
                audio_array = np.array(audio_array)
            
            # Compute duration
            duration = get_audio_duration(audio_array, sampling_rate)
            
            # Define audio file paths
            audio_filename = f"{dataset_name}_{idx:06d}.wav"
            local_audio_path = audio_dir / audio_filename
            
            # Save audio file
            if save_audio:
                try:
                    save_audio_file(audio_array, sampling_rate, str(local_audio_path))
                except Exception as e:
                    print(f"Warning: Failed to save audio for sample {idx}: {e}")
                    failed += 1
                    continue
            
            # Create manifest entry with relative path
            entry = create_manifest_entry(
                sample=sample,
                audio_filename=audio_filename,
                duration=duration,
                dataset_name=dataset_name,
                sample_id=idx,
                category=category,
            )
            
            manifest_entries.append(entry)
            successful += 1
            
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            failed += 1
            continue
    
    # Save dataset-specific manifest
    manifest_path = manifest_dir / f"{dataset_name}.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"✓ Saved {successful} samples to {manifest_path}")
    if failed > 0:
        print(f"✗ Failed to process {failed} samples")
    
    return successful, manifest_entries


def main():
    parser = argparse.ArgumentParser(
        description="Prepare AudioBench datasets for nemo-skills evaluation"
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to prepare",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (defaults to $NEMO_SKILLS_DATA_DIR/audiobench)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Specific dataset(s) to process (e.g., librispeech_test_clean earnings21)",
    )
    parser.add_argument(
        "--category",
        choices=["judge", "nonjudge", "all"],
        default="all",
        help="Process only judge, nonjudge, or all datasets",
    )
    parser.add_argument(
        "--no-audio",
        dest="save_audio",
        action="store_false",
        help="Skip saving audio files (only create manifests)",
    )
    parser.add_argument(
        "--audiobench-path",
        type=str,
        default=None,
        help="Path to AudioBench repository (will auto-clone if not found)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum number of samples to process per dataset (-1 for all)",
    )
    parser.set_defaults(save_audio=True)
    
    args = parser.parse_args()
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Use dataset directory as output (files will be in nemo_skills/dataset/audiobench/)
        output_dir = Path(__file__).parent
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine AudioBench repository path
    if args.audiobench_path:
        audiobench_path = Path(args.audiobench_path)
    else:
        audiobench_path = os.getenv("AUDIOBENCH_REPO_PATH")
        if audiobench_path:
            audiobench_path = Path(audiobench_path)
        else:
            # Default to AudioBench directory (same level as this script)
            audiobench_path = Path(__file__).parent / "AudioBench"
    
    # Clone AudioBench if it doesn't exist
    if not audiobench_path.exists():
        print(f"\nAudioBench not found at {audiobench_path}")
        if not clone_audiobench_repo(audiobench_path):
            print("\nFailed to clone AudioBench. Please clone it manually:")
            print("  git clone https://github.com/AudioLLMs/AudioBench.git")
            sys.exit(1)
    
    # Verify AudioBench structure
    if not (audiobench_path / "src" / "dataset.py").exists():
        print(f"\nError: AudioBench repository at {audiobench_path} is missing src/dataset.py")
        print("Please ensure the repository is properly cloned.")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("AudioBench Dataset Preparation")
    print("="*60)
    print(f"AudioBench path: {audiobench_path}")
    print(f"Output directory: {output_dir}")
    print(f"Save audio files: {args.save_audio}")
    print(f"Split: {args.split}")
    print("="*60 + "\n")
    
    # Determine which datasets to process
    if args.datasets:
        target_datasets = args.datasets
    else:
        all_datasets = JUDGE_DATASETS + NONJUDGE_DATASETS
        if args.category == "judge":
            target_datasets = JUDGE_DATASETS
        elif args.category == "nonjudge":
            target_datasets = NONJUDGE_DATASETS
        else:  # all
            target_datasets = all_datasets
    
    print(f"Processing {len(target_datasets)} dataset(s)\n")
    
    # Process datasets
    total_samples = 0
    successful_datasets = []
    failed_datasets = []
    judge_entries = []
    nonjudge_entries = []
    
    for dataset_name in target_datasets:
        try:
            num_samples, entries = process_dataset(
                dataset_name=dataset_name,
                output_dir=output_dir,
                save_audio=args.save_audio,
                split=args.split,
                audiobench_path=audiobench_path,
                max_samples=args.max_samples,
            )
            total_samples += num_samples
            successful_datasets.append(dataset_name)
            
            if dataset_name in JUDGE_DATASETS:
                judge_entries.extend(entries)
            elif dataset_name in NONJUDGE_DATASETS:
                nonjudge_entries.extend(entries)
                
        except Exception as e:
            print(f"\n✗ FAILED: {dataset_name}")
            print(f"  Error: {e}\n")
            failed_datasets.append((dataset_name, str(e)))
    
    # Create combined test.jsonl files
    if judge_entries:
        judge_test_path = output_dir / "judge" / f"{args.split}.jsonl"
        judge_test_path.parent.mkdir(parents=True, exist_ok=True)
        with open(judge_test_path, "w", encoding="utf-8") as f:
            for entry in judge_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\n✓ Combined judge {args.split}.jsonl: {judge_test_path}")
        print(f"  Total samples: {len(judge_entries)}")
    
    if nonjudge_entries:
        nonjudge_test_path = output_dir / "nonjudge" / f"{args.split}.jsonl"
        nonjudge_test_path.parent.mkdir(parents=True, exist_ok=True)
        with open(nonjudge_test_path, "w", encoding="utf-8") as f:
            for entry in nonjudge_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"\n✓ Combined nonjudge {args.split}.jsonl: {nonjudge_test_path}")
        print(f"  Total samples: {len(nonjudge_entries)}")
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Datasets requested: {len(target_datasets)}")
    print(f"Successfully processed: {len(successful_datasets)}")
    print(f"Failed: {len(failed_datasets)}")
    print(f"Total samples: {total_samples}")
    
    if successful_datasets:
        print(f"\nSuccessful datasets ({len(successful_datasets)}):")
        for name in successful_datasets:
            category = "judge" if name in JUDGE_DATASETS else "nonjudge"
            print(f"  ✓ {name} ({category})")
    
    if failed_datasets:
        print(f"\nFailed datasets ({len(failed_datasets)}):")
        for name, error in failed_datasets:
            print(f"  ✗ {name}: {error}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
