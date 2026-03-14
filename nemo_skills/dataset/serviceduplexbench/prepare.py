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
import shutil
from pathlib import Path

DEFAULT_DATA_SOURCE = (
    "/lustre/fs12/portfolios/llmservice/projects/llmservice_nemo_speechlm"
    "/data/nemo_skills/dataset/serviceduplexbench"
)


def format_entry(metadata, audio_rel_path):
    """Format a single sample into nemo-skills JSONL entry with OpenAI messages format.

    Each sample gets its own system prompt from metadata['prompt_text'].
    Creates three message variants:
    - messages: audio only (for speech-only evaluation)
    - messages_text_audio: both text and audio
    - messages_text: text only (for text-only comparison)
    """
    system_prompt = metadata["prompt_text"]
    question_text = metadata["question_text"]
    question_index = metadata["question_index"]
    speaker_index = metadata["speaker_index"]

    system_message = {"role": "system", "content": system_prompt}

    formatted = {
        "problem": question_text,
        "question_index": question_index,
        "speaker_index": speaker_index,
        "prompt_text": system_prompt,
    }

    audio_info = {"audio": {"path": audio_rel_path}}

    # 1. messages: audio only (empty content, with audio)
    user_message_audio = {"role": "user", "content": ""}
    user_message_audio.update(audio_info)
    formatted["messages"] = [system_message.copy(), user_message_audio]

    # 2. messages_text_audio: both text and audio
    user_message_text_audio = {"role": "user", "content": question_text}
    user_message_text_audio.update(audio_info)
    formatted["messages_text_audio"] = [system_message.copy(), user_message_text_audio]

    # 3. messages_text: text only (no audio)
    user_message_text = {"role": "user", "content": question_text}
    formatted["messages_text"] = [system_message.copy(), user_message_text]

    return formatted


def main():
    parser = argparse.ArgumentParser(description="Prepare ServiceDuplexBench dataset for nemo-skills")
    parser.add_argument(
        "--data_source",
        default=DEFAULT_DATA_SOURCE,
        help="Path to the source serviceduplexbench dataset directory",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        default=True,
        help="Use symlinks instead of copying audio files (default: True)",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy audio files instead of symlinking",
    )
    args = parser.parse_args()

    use_symlink = not args.copy

    data_source = Path(args.data_source)
    if not data_source.exists():
        raise FileNotFoundError(f"Data source not found: {data_source}")

    output_dir = Path(__file__).parent
    audio_dir = output_dir / "data"
    audio_dir.mkdir(parents=True, exist_ok=True)

    sample_dirs = sorted(
        [d for d in data_source.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda d: int(d.name),
    )

    if not sample_dirs:
        raise RuntimeError(f"No numbered subdirectories found in {data_source}")

    print(f"Found {len(sample_dirs)} samples in {data_source}")

    entries = []
    for sample_dir in sample_dirs:
        metadata_path = sample_dir / "metadata.json"
        audio_path = sample_dir / "input.wav"

        if not metadata_path.exists():
            print(f"Warning: skipping {sample_dir.name}, missing metadata.json")
            continue
        if not audio_path.exists():
            print(f"Warning: skipping {sample_dir.name}, missing input.wav")
            continue

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        idx = int(sample_dir.name)
        dest_audio = audio_dir / f"{idx}.wav"
        audio_rel_path = f"serviceduplexbench/data/{idx}.wav"

        if not dest_audio.exists():
            if use_symlink:
                os.symlink(str(audio_path.resolve()), str(dest_audio))
            else:
                shutil.copy2(str(audio_path), str(dest_audio))

        entry = format_entry(metadata, audio_rel_path)
        entries.append(entry)

    output_file = output_dir / "test.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")

    print(f"Wrote {len(entries)} entries to {output_file}")


if __name__ == "__main__":
    main()
