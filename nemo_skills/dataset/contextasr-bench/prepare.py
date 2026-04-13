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

Reads the ContextASR-Speech English JSONL and produces three split files
(one per evaluation mode: contextless, coarse, fine), each in OpenAI
message format with audio metadata.

Dataset: https://huggingface.co/datasets/MrSupW/ContextASR-Bench

Usage:
    ns prepare_data contextasr-bench --data_dir=/path/to/ContextASR-Bench

The --data_dir should point to the root of the ContextASR-Bench dataset
containing ContextASR-Speech_English.jsonl and the audio/ directory.
"""

import argparse
import json
from pathlib import Path


PROMPT_CONTEXTLESS = (
    "Transcribe the English audio into text, ensuring all punctuation marks are included."
)
PROMPT_COARSE = (
    "This audio belongs to the {domain_label} field. "
    "Transcribe the English audio into text, ensuring all punctuation marks are included."
)
PROMPT_FINE = (
    "This audio belongs to the {domain_label} field and may contains the following "
    "words or phrases: {entity_list}. "
    "Transcribe the English audio into text, ensuring all punctuation marks are included."
)


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
    """Format a single dataset sample into a JSONL record for a given mode.

    Args:
        sample: Raw dataset record with uniq_id, text, audio, domain_label, entity_list, duration.
        mode: One of "contextless", "coarse", "fine".
        audio_prefix: Base path prefix for audio files.
    """
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
    parser = argparse.ArgumentParser(description="Prepare ContextASR-Bench for NeMo Skills")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to ContextASR-Bench dataset root (contains ContextASR-Speech_English.jsonl and audio/)",
    )
    parser.add_argument(
        "--audio-prefix",
        type=str,
        default=None,
        help="Override audio path prefix in JSONL. Defaults to --data_dir value.",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip audio file verification (not recommended)",
    )
    args = parser.parse_args()

    output_dir = Path(__file__).parent

    data_dir = args.data_dir
    if data_dir is None:
        data_dir = str(output_dir)

    audio_prefix = args.audio_prefix if args.audio_prefix else data_dir
    audio_prefix = audio_prefix.rstrip("/")

    jsonl_path = Path(data_dir) / "ContextASR-Speech_English.jsonl"
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {jsonl_path}\n"
            f"Please download the ContextASR-Bench dataset and pass --data_dir pointing to its root."
        )

    print(f"Reading dataset from {jsonl_path}")
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

    modes = {
        "contextless": output_dir / "contextless" / "test.jsonl",
        "coarse": output_dir / "coarse" / "test.jsonl",
        "fine": output_dir / "fine" / "test.jsonl",
    }

    for mode_name, output_path in modes.items():
        output_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(output_path, "w", encoding="utf-8") as fout:
            for sample in samples:
                entry = format_entry(sample, mode_name, audio_prefix)
                fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
        print(f"  {mode_name}: wrote {count} samples to {output_path}")

    print(f"\nDone. Total: {len(samples)} samples x 3 modes = {len(samples) * 3} records")


if __name__ == "__main__":
    main()
