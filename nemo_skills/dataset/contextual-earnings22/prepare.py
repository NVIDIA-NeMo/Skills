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

"""Prepare Contextual Earnings-22 dataset for NeMo Skills evaluation.

Downloads the Argmax/Contextual Earnings-22 ``test`` split from HuggingFace
(``argmaxinc/contextual-earnings22``, ~404 MB total), extracts each clip's
audio array to a 16 kHz WAV file under ``<data_dir>/audio/``, and produces
three JSONL split files - one per evaluation mode (contextless, local, global).

If ``--data_dir`` is provided and already contains the prepared audio, the
download/extract step is skipped and the existing data is used directly.

Dataset: https://huggingface.co/datasets/argmaxinc/contextual-earnings22
Paper:   Contextual Earnings-22 (Argmax, 2025) -- https://arxiv.org/abs/2604.07354

Usage:
    # Auto-download to a sibling directory of the repo
    ns prepare_data contextual-earnings22

    # Use pre-downloaded / cached data
    ns prepare_data contextual-earnings22 --data_dir=/path/to/contextual-earnings22

    # Override audio path prefix written into JSONL (e.g. for container mounts)
    ns prepare_data contextual-earnings22 --data_dir=/path --audio-prefix /data/ce22
"""

import argparse
import json
import wave
from pathlib import Path

import numpy as np

HF_REPO_ID = "argmaxinc/contextual-earnings22"
BENCHMARK_NAME = "contextual-earnings22"
SPLIT = "test"
SAMPLING_RATE = 16000

PROMPT_CONTEXTLESS = "Transcribe the English audio into text, ensuring all punctuation marks are included."
PROMPT_LOCAL = (
    "This audio may contain the following words or phrases: {keyword_list}. "
    "Transcribe the English audio into text, ensuring all punctuation marks are included."
)
PROMPT_GLOBAL = (
    "This audio may contain the following words or phrases (some may not be present in the clip): "
    "{keyword_list}. "
    "Transcribe the English audio into text, ensuring all punctuation marks are included."
)


def _default_data_dir() -> Path:
    """Pick a dataset download location that sits OUTSIDE the repo.

    Walks up from this file until it finds the repo root (identified by
    ``pyproject.toml``) and returns ``<repo_root>.parent / "<repo_name>-data" /
    <BENCHMARK_NAME>``. Falls back to ``~/.cache/nemo-skills-data/<bench>`` if
    no repo root is found. Never returns a path inside the repo tree.
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent.parent / f"{parent.name}-data" / BENCHMARK_NAME
    return Path.home() / ".cache" / "nemo-skills-data" / BENCHMARK_NAME


def _audio_subdir(data_dir: Path) -> Path:
    return data_dir / "audio"


def _audio_filename(file_id: str, audio_start: float, idx: int) -> str:
    """Stable, unique audio filename for a clip.

    ``file_id + audio_start`` uniquely identifies a clip in the source dataset
    (multiple clips can share a ``file_id`` since they are 15 s windows from
    the same call). We include ``idx`` only as a deterministic tiebreaker for
    duplicates at the same start position.
    """
    return f"{file_id}__{audio_start:09.2f}__{idx}.wav"


def _write_wav(path: Path, samples: np.ndarray, sampling_rate: int) -> None:
    """Write a mono int16 WAV at ``sampling_rate`` Hz."""
    if samples.ndim > 1:
        samples = samples.mean(axis=tuple(range(1, samples.ndim)))
    if np.issubdtype(samples.dtype, np.floating):
        samples = np.clip(samples, -1.0, 1.0)
        samples_i16 = (samples * 32767.0).astype(np.int16)
    else:
        samples_i16 = samples.astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sampling_rate)
        wf.writeframes(samples_i16.tobytes())


def download_and_extract_audio(data_dir: Path) -> int:
    """Download the HF parquet split and write per-clip WAV files.

    Returns the number of clips written. Skips clips that already exist on disk.
    """
    from datasets import load_dataset

    audio_dir = _audio_subdir(data_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Loading {HF_REPO_ID} (split={SPLIT}) from HuggingFace")
    print(f"Destination: {data_dir}")
    print("Total download size: ~404 MB (parquet with 16 kHz embedded audio)")
    print("=" * 70)

    ds = load_dataset(HF_REPO_ID, split=SPLIT)

    print(f"Loaded {len(ds)} samples. Extracting audio to {audio_dir} ...")

    samples = []
    written = 0
    for idx, row in enumerate(ds):
        audio = row["audio"]
        file_id = row["file_id"]
        audio_start = float(row["audio_start"])
        wav_name = _audio_filename(file_id, audio_start, idx)
        wav_path = audio_dir / wav_name

        if not wav_path.exists():
            arr = np.asarray(audio["array"])
            sr = int(audio["sampling_rate"])
            if sr != SAMPLING_RATE:
                raise ValueError(
                    f"Unexpected sampling rate {sr} for sample {idx} (file_id={file_id}); expected {SAMPLING_RATE}."
                )
            _write_wav(wav_path, arr, sr)
            written += 1

        samples.append(
            {
                "file_id": file_id,
                "audio_filename": wav_name,
                "audio_start": audio_start,
                "audio_end": float(row["audio_end"]),
                "duration": float(row["duration"]),
                "text": row["text"],
                "norm_text": row["norm_text"],
                "transcript_tokens": list(row["transcript"]),
                "local_keywords": list(row["dictionary"]),
                "global_keywords": list(row["keywords"]),
                "is_flagged": bool(row["is_flagged"]),
            }
        )

    print(f"Wrote {written} new audio files (skipped {len(samples) - written} already present).")

    metadata_path = data_dir / "samples.jsonl"
    with open(metadata_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"Wrote sample metadata: {metadata_path}")

    return len(samples)


def load_samples(data_dir: Path) -> list[dict]:
    """Load the cached sample metadata produced by ``download_and_extract_audio``."""
    metadata_path = data_dir / "samples.jsonl"
    samples: list[dict] = []
    with open(metadata_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def build_messages(prompt_text: str, audio_path: str, duration: float) -> list[dict]:
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


def format_entry(sample: dict, mode: str, audio_prefix: str) -> dict:
    """Format a single dataset sample into a JSONL record for a given mode."""
    audio_path = f"{audio_prefix}/audio/{sample['audio_filename']}"
    local_keywords = sample["local_keywords"]
    global_keywords = sample["global_keywords"]

    if mode == "contextless":
        prompt = PROMPT_CONTEXTLESS
        eval_keywords = local_keywords
    elif mode == "local":
        prompt = PROMPT_LOCAL.format(keyword_list=", ".join(local_keywords))
        eval_keywords = local_keywords
    elif mode == "global":
        prompt = PROMPT_GLOBAL.format(keyword_list=", ".join(global_keywords))
        eval_keywords = local_keywords
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {
        "messages": build_messages(prompt, audio_path, sample["duration"]),
        "expected_answer": sample["norm_text"],
        "expected_answer_cased": sample["text"],
        "keyword_list": eval_keywords,
        "local_keywords": local_keywords,
        "global_keywords": global_keywords,
        "subset_for_metrics": sample["file_id"],
        "file_id": sample["file_id"],
        "audio_start": sample["audio_start"],
        "duration": float(sample["duration"]),
        "audio_filepath": audio_path,
    }


def main() -> None:
    """Parse arguments, download/extract data if needed, and write per-mode JSONL splits."""
    parser = argparse.ArgumentParser(description="Prepare Contextual Earnings-22 for NeMo Skills")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=(
            "Path to Contextual-Earnings-22 dataset root. If the directory already "
            "contains audio/ and samples.jsonl, that data is used directly. If the "
            "directory is empty or incomplete, data will be downloaded there "
            "automatically from HuggingFace. If not provided, defaults to a sibling "
            "directory of the repo (never inside the repo tree)."
        ),
    )
    parser.add_argument(
        "--audio-prefix",
        type=str,
        default=None,
        help=(
            "Override audio path prefix written into JSONL files. Defaults to the "
            "data_dir value. Useful for container mount points "
            "(e.g., --audio-prefix /data/contextual-earnings22)."
        ),
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip downloading/verifying audio files (not recommended for actual evaluation).",
    )
    args = parser.parse_args()

    output_dir = Path(__file__).parent
    data_dir = Path(args.data_dir) if args.data_dir else _default_data_dir()
    metadata_path = data_dir / "samples.jsonl"

    if metadata_path.exists():
        print(f"Using pre-downloaded data from {data_dir}")
    elif args.no_audio:
        raise FileNotFoundError(
            f"Sample metadata not found at {metadata_path}.\n"
            f"Cannot use --no-audio when data has not been downloaded yet. "
            f"Either run without --no-audio first to download, or point --data_dir "
            f"to a directory that already contains samples.jsonl."
        )
    else:
        print(f"Data not found at {data_dir}. Downloading there...")
        download_and_extract_audio(data_dir)

    audio_prefix = args.audio_prefix if args.audio_prefix else str(data_dir)
    audio_prefix = audio_prefix.rstrip("/")

    print(f"\nReading sample metadata from {metadata_path}")
    samples = load_samples(data_dir)
    print(f"Loaded {len(samples)} samples")
    if not samples:
        raise ValueError(
            f"No samples found in {metadata_path}. Re-run without --no-audio to regenerate dataset metadata."
        )

    if not args.no_audio:
        sample_audio = Path(audio_prefix) / "audio" / samples[0]["audio_filename"]
        if not sample_audio.exists():
            print(
                f"WARNING: Sample audio file not found at {sample_audio}. "
                f"Audio paths may need adjustment via --audio-prefix."
            )
        else:
            print(f"Audio files verified (sample check: {sample_audio})")

    modes = {
        "contextless": output_dir / "contextless" / "test.jsonl",
        "local": output_dir / "local" / "test.jsonl",
        "global": output_dir / "global" / "test.jsonl",
    }

    print("\nWriting JSONL splits...")
    for mode_name, output_path in modes.items():
        lines = [json.dumps(format_entry(s, mode_name, audio_prefix), ensure_ascii=False) for s in samples]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fout:
            for line in lines:
                fout.write(line + "\n")
        print(f"  {mode_name}: wrote {len(lines)} samples to {output_path}")

    print(f"\nDone. Total: {len(samples)} samples x 3 modes = {len(samples) * 3} records")


if __name__ == "__main__":
    main()
