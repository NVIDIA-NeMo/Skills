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

import argparse
import json
import os
import subprocess
import zipfile
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from nemo_skills.dataset.utils import build_container_audio_path, get_container_audio_root, get_mcq_fields


def download_mmau_data(download_dir, hf_token):
    """Download and extract MMAU-Pro data.zip file."""
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    data_zip_path = download_dir / "data.zip"
    extracted_data_dir = download_dir / "data"

    if extracted_data_dir.exists() and any(extracted_data_dir.iterdir()):
        print(f"Data already exists at {extracted_data_dir}")
        return extracted_data_dir

    if not data_zip_path.exists():
        print("Downloading MMAU-Pro data.zip...")
        cmd = [
            "wget",
            "--header",
            f"Authorization: Bearer {hf_token}",
            "https://huggingface.co/datasets/gamma-lab-umd/MMAU-Pro/resolve/main/data.zip",
            "-O",
            str(data_zip_path),
        ]
        subprocess.run(cmd, check=True)

    print("Extracting data.zip...")
    with zipfile.ZipFile(data_zip_path, "r") as zip_ref:
        zip_ref.extractall(download_dir)

    return extracted_data_dir


def _normalize_audio_path(path: str, audio_root: str) -> str:
    """Convert a dataset audio path into the configured in-container path.

    Preserves the dataset's relative layout (e.g. ``data/<uuid>.wav``) verbatim
    so the JSONL path matches where files actually live after ``cp -r``. The
    HF MMAU-Pro dataset records audio entries as ``data/<uuid>.wav`` and that
    ``data/`` is the subdir its zip extracts into; stripping it here would
    produce JSONL paths that fail to resolve against the on-disk layout.
    """
    rel_path = str(path).lstrip("/")
    return build_container_audio_path("mmau-pro", rel_path, audio_prefix=audio_root)


def format_entry(entry, with_audio=False, audio_root="/data"):
    """Format entry for nemo-skills with OpenAI messages and audio support."""
    choices = entry.get("choices", []) or []

    formatted_entry = {
        "expected_answer": entry["answer"],
        **get_mcq_fields(entry["question"], choices),
        **{k: v for k, v in entry.items() if k not in ["answer"]},
    }

    category = entry.get("category", "")

    # Add subset_for_metrics for closed-form questions to track different domains
    if category not in ["open", "instruction following"]:
        formatted_entry["subset_for_metrics"] = category

    if category == "open":
        content = entry["question"]
    elif choices and len(choices) > 1:
        options_text = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))
        content = f"{entry['question']}\n\n{options_text}"
    else:
        content = entry["question"]

    user_message = {"role": "user", "content": content}

    if entry.get("audio_path"):
        audio_path = entry["audio_path"]

        if isinstance(audio_path, list) and audio_path:
            audio_paths = [_normalize_audio_path(path, audio_root) for path in audio_path]
            formatted_entry["audio_path"] = audio_paths
            user_message["audios"] = [{"path": path, "duration": 10.0} for path in audio_paths]
        elif isinstance(audio_path, str):
            audio_path = _normalize_audio_path(audio_path, audio_root)
            formatted_entry["audio_path"] = audio_path
            user_message["audio"] = {"path": audio_path, "duration": 10.0}

    formatted_entry["messages"] = [user_message]
    return formatted_entry


def main():
    parser = argparse.ArgumentParser(description="Prepare MMAU-Pro dataset for nemo-skills")
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip downloading audio files",
    )
    parser.add_argument(
        "--audio-prefix",
        type=str,
        default=None,
        help="In-container audio root written into JSONL paths. Defaults to $NEMO_SKILLS_AUDIO_ROOT or /data.",
    )
    args = parser.parse_args()

    data_dir = Path(__file__).parent
    audio_root = get_container_audio_root(args.audio_prefix)
    print(f"Audio paths in JSONL will use: {audio_root}/mmau-pro/...")

    # Download audio by default unless --no-audio is specified
    if not args.no_audio:
        if not os.environ.get("HF_TOKEN"):
            raise ValueError(
                "HF_TOKEN environment variable required for audio download. Use --no-audio to skip (not recommended)."
            )
        download_mmau_data(data_dir, os.environ["HF_TOKEN"])

    print(f"Loading {args.split} split...")
    dataset = load_dataset("gamma-lab-umd/MMAU-Pro", trust_remote_code=True)[args.split]

    # Separate files for each evaluation category
    category_files = {
        "closed_form": data_dir / "closed_form" / f"{args.split}.jsonl",
        "open": data_dir / "open_ended" / f"{args.split}.jsonl",
        "instruction following": data_dir / "instruction_following" / f"{args.split}.jsonl",
    }

    for category_file in category_files.values():
        category_file.parent.mkdir(parents=True, exist_ok=True)

    category_file_handles = {
        category: open(file_path, "w", encoding="utf-8") for category, file_path in category_files.items()
    }

    print(f"Processing {len(dataset)} entries into separate category files...")
    category_counts = {category: 0 for category in category_files.keys()}

    try:
        for entry in tqdm(dataset):
            formatted_entry = format_entry(entry, with_audio=not args.no_audio, audio_root=audio_root)
            category = entry.get("category", "closed_form")

            # Map category to file
            if category == "instruction following":
                target_category = "instruction following"
            elif category == "open":
                target_category = "open"
            else:
                # All other categories (closed-form, multiple choice) go to closed_form
                target_category = "closed_form"

            category_file_handles[target_category].write(json.dumps(formatted_entry) + "\n")
            category_counts[target_category] += 1
    finally:
        for fh in category_file_handles.values():
            fh.close()

    print("Dataset split into categories:")
    for category, file_path in category_files.items():
        print(f"  - {category}: {category_counts[category]} entries -> {file_path}")


if __name__ == "__main__":
    main()
