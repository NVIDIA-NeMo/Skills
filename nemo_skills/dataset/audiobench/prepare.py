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
    python -m nemo_skills.dataset.audiobench.prepare --datasets librispeech_test_clean earnings21_test
    python -m nemo_skills.dataset.audiobench.prepare --category nonjudge
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List

import numpy as np
import soundfile as sf
from tqdm import tqdm

# AudioBench datasets categorized by evaluation type
JUDGE_DATASETS = [
    "alpaca_audio_test",
    "audiocaps_qa_test",
    "audiocaps_test",
    "clotho_aqa_test",
    "cn_college_listen_mcq_test",
    "dream_tts_mcq_test",
    "iemocap_emotion_test",
    "iemocap_gender_test",
    "imda_ar_dialogue",
    "imda_ar_sentence",
    "imda_gr_dialogue",
    "imda_gr_sentence",
    "imda_part3_30s_ds_human_test",
    "imda_part4_30s_ds_human_test",
    "imda_part5_30s_ds_human_test",
    "imda_part6_30s_ds_human_test",
    "imda_part3_30s_sqa_human_test",
    "imda_part4_30s_sqa_human_test",
    "imda_part5_30s_sqa_human_test",
    "imda_part6_30s_sqa_human_test",
    "meld_emotion_test",
    "meld_sentiment_test",
    "mmau_mini",
    "muchomusic_test",
    "openhermes_audio_test",
    "public_sg_speech_qa_test",
    "slue_p2_sqa5_test",
    "spoken_squad_test",
    "voxceleb_accent_test",
    "voxceleb_gender_test",
    "wavcaps_qa_test",
    "wavcaps_test",
]

NONJUDGE_DATASETS = [
    "aishell_asr_zh_test",
    "common_voice_15_en_test",
    "covost2_en_id_test",
    "covost2_en_ta_test",
    "covost2_en_zh_test",
    "covost2_id_en_test",
    "covost2_ta_en_test",
    "covost2_zh_en_test",
    "earnings21_test",
    "earnings22_test",
    "gigaspeech_test",
    "gigaspeech2_indo",
    "gigaspeech2_thai",
    "gigaspeech2_viet",
    "imda_part1_asr_test",
    "imda_part2_asr_test",
    "imda_part3_30s_asr_test",
    "imda_part4_30s_asr_test",
    "imda_part5_30s_asr_test",
    "imda_part6_30s_asr_test",
    "librispeech_test_clean",
    "librispeech_test_other",
    "peoples_speech_test",
    "seame_dev_man",
    "seame_dev_sge",
    "spoken-mqa_long_digit",
    "spoken-mqa_multi_step_reasoning",
    "spoken-mqa_short_digit",
    "spoken-mqa_single_step_reasoning",
    "tedlium3_test",
    "tedlium3_long_form_test",
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


def extract_audio_dict(sample: Dict) -> Dict | None:
    """Extract an Audio feature dict from a HuggingFace sample.

    AudioLLMs-hosted AudioBench datasets commonly store audio under the `context`
    column (HF Audio feature), while other sources may use `audio`.
    """
    # Prefer official HF Audio feature columns if present
    for key in ("context", "audio"):
        audio_dict = sample.get(key)
        if isinstance(audio_dict, dict):
            return audio_dict
    return None


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
    instruction = sample.get("instruction", sample.get("text", "Process the audio"))
    reference = sample.get("reference", sample.get("answer", ""))
    task_type = sample.get("task_type", "unknown")

    # Create absolute audio path with /data/ prefix for cluster deployment
    # Format: /data/audiobench/{category}/audio/{dataset_name}/{filename}
    audio_rel_path = f"/data/audiobench/{category}/audio/{dataset_name}/{audio_filename}"

    # Create audio metadata (both singular and plural forms for compatibility)
    audio_metadata = {"path": audio_rel_path, "duration": duration}

    entry = {
        "expected_answer": reference,
        "audio_path": [audio_rel_path],
        # Used by audio metrics to decide whether to parse LLM-as-a-judge results.
        # AudioBench "judge" datasets are open-ended (judged), while "nonjudge" datasets are closed-form.
        "category": "open" if category == "judge" else ("closed" if category == "nonjudge" else category),
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. /no_think"},
            {
                "role": "user",
                "content": instruction,
                "audio": audio_metadata,
                "audios": [audio_metadata],
            },
        ],
        "dataset": dataset_name,
        "subset_for_metrics": dataset_name,
        "sample_id": sample_id,
        "task_type": task_type,
        "question": instruction,
    }

    for key in [
        "choices",
        "options",
        "audio_text_instruction",
        "audio_gt",
        "dimension",
        "rule_type",
        "rule_target",
        "task",
    ]:
        if key in sample:
            entry[key] = sample[key]

    return entry


def process_dataset(
    dataset_name: str,
    output_dir: Path,
    save_audio: bool = True,
    split: str = "test",
    max_samples: int = -1,
) -> tuple[int, List[Dict]]:
    """Process a single AudioBench dataset.

    Args:
        dataset_name: Name of the dataset to process
        output_dir: Base output directory
        save_audio: Whether to save audio files
        split: Dataset split (default: "test")
        max_samples: Max number of samples to process (-1 for all)

    Returns:
        Tuple of (num_samples, manifest_entries)
    """
    print(f"\n{'=' * 60}")
    print(f"Processing: {dataset_name}")
    print(f"{'=' * 60}")

    try:
        from datasets import load_dataset
    except Exception as e:
        raise ImportError(
            f"Failed to import HuggingFace 'datasets'. Please ensure it is installed.\nOriginal error: {e}"
        )

    # Upstream reference: https://github.com/AudioLLMs/AudioBench
    try:
        # AudioBench mapping for datasets that are not 1:1 AudioLLMs/<dataset_name>.
        hf_map = {
            # AudioLLMs org aliases
            "aishell_asr_zh_test": {"repo": "AudioLLMs/aishell_1_zh_test", "split": "test"},
            "muchomusic_test": {"repo": "AudioLLMs/mu_chomusic_test", "split": "test"},
            "openhermes_audio_test": {"repo": "AudioLLMs/openhermes_instruction_test", "split": "test"},
            "iemocap_emotion_test": {"repo": "AudioLLMs/iemocap_emotion_recognition", "split": "test"},
            "iemocap_gender_test": {"repo": "AudioLLMs/iemocap_gender_recognition", "split": "test"},
            "mmau_mini": {
                "repo": "AudioLLMs/MMAU-mini",
                "split": "test",
                "fallback_repo": "AudioLLMs/MMAU-mini-do-not-use",
            },
            # GigaSpeech2 variants (one repo with data_dir selector)
            "gigaspeech2_thai": {"repo": "AudioLLMs/gigaspeech2-test", "split": "train", "data_dir": "th-test"},
            "gigaspeech2_indo": {"repo": "AudioLLMs/gigaspeech2-test", "split": "train", "data_dir": "id-test"},
            "gigaspeech2_viet": {"repo": "AudioLLMs/gigaspeech2-test", "split": "train", "data_dir": "vi-test"},
            "spoken-mqa_short_digit": {"repo": "amao0o0/spoken-mqa", "split": "short_digit"},
            "spoken-mqa_long_digit": {"repo": "amao0o0/spoken-mqa", "split": "long_digit"},
            "spoken-mqa_single_step_reasoning": {"repo": "amao0o0/spoken-mqa", "split": "single_step_reasoning"},
            "spoken-mqa_multi_step_reasoning": {"repo": "amao0o0/spoken-mqa", "split": "multi_step_reasoning"},
            "imda_part1_asr_test": {
                "repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",
                "split": "train",
                "data_dir": "ASR-PART1-Test",
            },
            "imda_part2_asr_test": {
                "repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",
                "split": "train",
                "data_dir": "ASR-PART2-Test",
            },
            "imda_part3_30s_asr_test": {
                "repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",
                "split": "train",
                "data_dir": "ASR-PART3-Test",
            },
            "imda_part4_30s_asr_test": {
                "repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",
                "split": "train",
                "data_dir": "ASR-PART4-Test",
            },
            "imda_part5_30s_asr_test": {
                "repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",
                "split": "train",
                "data_dir": "ASR-PART5-Test",
            },
            "imda_part6_30s_asr_test": {
                "repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",
                "split": "train",
                "data_dir": "ASR-PART6-Test",
            },
            "imda_part3_30s_sqa_human_test": {
                "repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",
                "split": "train",
                "data_dir": "SQA-PART3-Test",
            },
            "imda_part4_30s_sqa_human_test": {
                "repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",
                "split": "train",
                "data_dir": "SQA-PART4-Test",
            },
            "imda_part5_30s_sqa_human_test": {
                "repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",
                "split": "train",
                "data_dir": "SQA-PART5-Test",
            },
            "imda_part6_30s_sqa_human_test": {
                "repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",
                "split": "train",
                "data_dir": "SQA-PART6-Test",
            },
            "imda_part3_30s_ds_human_test": {
                "repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",
                "split": "train",
                "data_dir": "SDS-PART3-Test",
            },
            "imda_part4_30s_ds_human_test": {
                "repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",
                "split": "train",
                "data_dir": "SDS-PART4-Test",
            },
            "imda_part5_30s_ds_human_test": {
                "repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",
                "split": "train",
                "data_dir": "SDS-PART5-Test",
            },
            "imda_part6_30s_ds_human_test": {
                "repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",
                "split": "train",
                "data_dir": "SDS-PART6-Test",
            },
            "imda_ar_sentence": {
                "repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",
                "split": "train",
                "data_dir": "PQA-AR-Sentence-Test",
            },
            "imda_ar_dialogue": {
                "repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",
                "split": "train",
                "data_dir": "PQA-AR-Dialogue-Test",
            },
            "imda_gr_sentence": {
                "repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",
                "split": "train",
                "data_dir": "PQA-GR-Sentence-Test",
            },
            "imda_gr_dialogue": {
                "repo": "MERaLiON/Multitask-National-Speech-Corpus-v1",
                "split": "train",
                "data_dir": "PQA-GR-Dialogue-Test",
            },
        }

        spec = hf_map.get(dataset_name)
        if spec is None:
            hf_repo = f"AudioLLMs/{dataset_name}"
            hf_split = split
            hf_ds = load_dataset(hf_repo, split=hf_split)
        else:
            hf_repo = spec["repo"]
            hf_split = spec.get("split", split)
            data_dir = spec.get("data_dir")
            if data_dir:
                hf_ds = load_dataset(hf_repo, data_dir=data_dir, split=hf_split)
            else:
                hf_ds = load_dataset(hf_repo, split=hf_split)

            fallback_repo = spec.get("fallback_repo")
            if fallback_repo:
                # Only try fallback if the primary repo is missing/inaccessible.
                # (Keep behavior deterministic and close to upstream mapping.)
                try:
                    _ = len(hf_ds)
                except Exception:
                    hf_repo = fallback_repo
                    hf_ds = load_dataset(hf_repo, split=hf_split)

        if max_samples is not None and int(max_samples) > 0:
            hf_ds = hf_ds.select(range(min(int(max_samples), len(hf_ds))))
        data_samples = hf_ds
        print(f"Loaded {len(hf_ds)} samples via HuggingFace datasets: {hf_repo} (split={hf_split})")
    except Exception as e:
        raise Exception(
            "Failed to load AudioBench dataset via HuggingFace.\n"
            f"- Requested dataset_name: {dataset_name}\n"
            f"- HuggingFace dataset repo attempted: {locals().get('hf_repo', 'UNKNOWN')}\n"
            f"- Split: {locals().get('hf_split', split)}\n"
            "Please verify the dataset exists under the AudioLLMs org:\n"
            "  https://huggingface.co/AudioLLMs/datasets\n"
            f"Original error: {e}"
        )

    # Determine category
    dataset_base = dataset_name.replace("_test", "")
    if dataset_name in JUDGE_DATASETS or dataset_base in JUDGE_DATASETS:
        category = "judge"
    elif dataset_name in NONJUDGE_DATASETS or dataset_base in NONJUDGE_DATASETS:
        category = "nonjudge"
    else:
        category = "unknown"

    # Output directories
    audio_dir = output_dir / category / "audio" / dataset_name
    dataset_dir = output_dir / category / dataset_name
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)

    # Copy __init__.py from category folder to dataset folder
    category_init = output_dir / category / "__init__.py"
    dataset_init = dataset_dir / "__init__.py"
    if category_init.exists() and not dataset_init.exists():
        shutil.copy2(category_init, dataset_init)
        print(f"✓ Copied __init__.py to {dataset_dir}")

    manifest_entries = []
    successful = 0
    failed = 0

    for idx, sample in enumerate(tqdm(data_samples, desc=f"Processing {dataset_name}")):
        try:
            # Get audio data
            audio_dict = extract_audio_dict(sample)
            if audio_dict is None:
                print(f"Warning: Sample {idx} has no audio, skipping")
                failed += 1
                continue

            # Extract audio array and sampling rate
            audio_array = audio_dict.get("array")
            sampling_rate = audio_dict.get("sampling_rate", 16000)

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

    # Save dataset-specific manifest to dataset directory
    manifest_path = dataset_dir / f"{split}.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in manifest_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✓ Saved {successful} samples to {manifest_path}")
    if failed > 0:
        print(f"✗ Failed to process {failed} samples")

    return successful, manifest_entries


def main():
    parser = argparse.ArgumentParser(description="Prepare AudioBench datasets for nemo-skills evaluation")
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

    print("\n" + "=" * 60)
    print("AudioBench Dataset Preparation")
    print("=" * 60)
    print("AudioBench source: HuggingFace datasets (AudioLLMs/AudioBench)")
    print(f"Output directory: {output_dir}")
    print(f"Save audio files: {args.save_audio}")
    print(f"Split: {args.split}")
    print("=" * 60 + "\n")

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

    # Initialize category folders with __init__.py for nemo-skills to find dataset defaults
    for category in ["judge", "nonjudge"]:
        category_dir = output_dir / category
        category_dir.mkdir(exist_ok=True)

        # Copy category __init__.py
        init_file = category_dir / "__init__.py"
        template_init = output_dir / category / "__init__.py"
        if not init_file.exists() and template_init.exists():
            shutil.copy2(template_init, init_file)

    total_samples = 0
    total_datasets = 0

    for name in target_datasets:
        # Normalize dataset name: allow passing without _test suffix
        dataset_name = name
        if dataset_name not in JUDGE_DATASETS and dataset_name not in NONJUDGE_DATASETS:
            # Try adding _test suffix (AudioBench uses mixed naming)
            if f"{dataset_name}_test" in JUDGE_DATASETS or f"{dataset_name}_test" in NONJUDGE_DATASETS:
                dataset_name = f"{dataset_name}_test"

        # Determine category for logging
        category = "judge" if name in JUDGE_DATASETS else "nonjudge"

        try:
            num_samples, _ = process_dataset(
                dataset_name=dataset_name,
                output_dir=output_dir,
                save_audio=args.save_audio,
                split=args.split,
                max_samples=args.max_samples,
            )
            total_samples += num_samples
            total_datasets += 1
            print(f"✓ Completed {dataset_name}: {num_samples} samples")
        except Exception as e:
            print(f"✗ Failed {dataset_name}: {e}")
            continue

    print("\n" + "=" * 60)
    print("AudioBench Preparation Summary")
    print("=" * 60)
    print(f"Datasets processed: {total_datasets}/{len(target_datasets)}")
    print(f"Total samples: {total_samples}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
