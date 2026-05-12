# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Prepare LibriSpeechMix SOT manifests for NeMo Skills."""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np

BENCHMARK_NAME = "librispeechmix-sot"
DURATION_THRESHOLD_SECONDS = 20.0
DEFAULT_PROMPT = "Transcribe the mixed audio with speaker tags like [s0] hello [s1] yes."
SPLITS = ("test-clean", "dev-clean")
MIXES = ("1mix", "2mix", "3mix")

LIST_URL = "https://raw.githubusercontent.com/NaoyukiKanda/LibriSpeechMix/master/list/{name}.jsonl"
OPENSLR_ARCHIVES = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
}


def _default_data_dir() -> Path:
    """Return a repo-sibling data directory, never an in-repo path."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent.parent / f"{parent.name}-data" / BENCHMARK_NAME
    return Path.home() / ".cache" / "nemo-skills-data" / BENCHMARK_NAME


def _resolve_values(values: list[str] | None, allowed: tuple[str, ...]) -> tuple[str, ...]:
    if values is None:
        return allowed
    out = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if not part:
                continue
            if part == "all":
                out.extend(allowed)
            elif part not in allowed:
                raise ValueError(f"Unsupported value {part!r}; expected one of {allowed}")
            else:
                out.append(part)
    return tuple(dict.fromkeys(out))


def _safe_extract_archive(archive_path: Path, output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (output_dir / member.name).resolve()
            if member_path != output_dir and output_dir not in member_path.parents:
                raise RuntimeError(f"Refusing to extract archive member outside target: {member.name}")
        tar.extractall(output_dir)


def ensure_librispeech_audio(split: str, data_dir: Path, no_audio: bool) -> Path:
    """Ensure the OpenSLR LibriSpeech split exists and return the LibriSpeech root."""
    librispeech_root = data_dir / "LibriSpeech"
    if (librispeech_root / split).exists():
        return librispeech_root
    if no_audio:
        return librispeech_root

    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / f"{split}.tar.gz"
    print("=" * 72)
    print(f"Downloading LibriSpeech {split} from OpenSLR into {data_dir}")
    print("This archive is large; pass --no-audio to prepare manifests without audio.")
    print("=" * 72)
    urllib.request.urlretrieve(OPENSLR_ARCHIVES[split], archive_path)
    _safe_extract_archive(archive_path, data_dir)
    archive_path.unlink(missing_ok=True)
    return librispeech_root


def ensure_librispeechmix_list(name: str, data_dir: Path, librispeechmix_dir: Path | None) -> Path:
    """Find or download an official LibriSpeechMix JSONL list file."""
    if librispeechmix_dir is not None:
        candidates = [librispeechmix_dir / "list" / f"{name}.jsonl", librispeechmix_dir / f"{name}.jsonl"]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not find {name}.jsonl under {librispeechmix_dir}")

    env_dir = os.getenv("LIBRISPEECHMIX_DIR")
    if env_dir:
        return ensure_librispeechmix_list(name, data_dir, Path(env_dir).expanduser())

    list_dir = data_dir / "LibriSpeechMix" / "list"
    list_dir.mkdir(parents=True, exist_ok=True)
    list_path = list_dir / f"{name}.jsonl"
    if list_path.exists():
        return list_path

    print(f"Downloading official LibriSpeechMix list {name}.jsonl")
    urllib.request.urlretrieve(LIST_URL.format(name=name), list_path)
    return list_path


def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file."""
    rows = []
    with open(path, "rt", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mixed_duration(row: dict) -> float:
    """Return mixed utterance duration from source delays and durations."""
    return float(max(delay + duration for delay, duration in zip(row["delays"], row["durations"])))


def duration_bucket(duration: float) -> str:
    """Map a duration to the explicit under/over-20 second benchmark split."""
    return "under20s" if duration <= DURATION_THRESHOLD_SECONDS else "over20s"


def speaker_tag_map(row: dict) -> dict[str, str]:
    """Assign stable [sN] tags in first-start order."""
    ordered = sorted(enumerate(row["speakers"]), key=lambda item: (row["delays"][item[0]], item[1], item[0]))
    mapping = {}
    for _orig_idx, speaker in ordered:
        if speaker not in mapping:
            mapping[speaker] = f"s{len(mapping)}"
    return mapping


def sot_text(row: dict) -> str:
    """Create SOT reference text with bracket speaker tags."""
    tag_by_speaker = speaker_tag_map(row)
    ordered = sorted(range(len(row["speakers"])), key=lambda idx: (row["delays"][idx], row["speakers"][idx], idx))
    chunks = []
    for idx in ordered:
        tag = tag_by_speaker[row["speakers"][idx]]
        chunks.append(f"[{tag}] {row['texts'][idx].lower()}")
    return " ".join(chunks)


def num_speaker_changes(row: dict) -> int:
    """Count speaker changes in chronological segment order."""
    ordered_speakers = [
        row["speakers"][idx]
        for idx in sorted(range(len(row["speakers"])), key=lambda i: (row["delays"][i], row["speakers"][i], i))
    ]
    return sum(1 for prev, cur in zip(ordered_speakers, ordered_speakers[1:]) if prev != cur)


def rttm_lines(row: dict, session_id: str) -> list[str]:
    """Render RTTM speaker supervision for one mixed recording."""
    lines = []
    for delay, duration, speaker in zip(row["delays"], row["durations"], row["speakers"]):
        lines.append(f"SPEAKER {session_id} 1 {delay:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n")
    return lines


def seglst_entries(row: dict, session_id: str) -> list[dict]:
    """Render reference SegLST entries for one mixed recording."""
    entries = []
    for delay, duration, speaker, text in zip(row["delays"], row["durations"], row["speakers"], row["texts"]):
        entries.append(
            {
                "session_id": session_id,
                "words": text.lower(),
                "speaker": speaker,
                "start_time": round(float(delay), 3),
                "end_time": round(float(delay + duration), 3),
            }
        )
    return entries


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sample_rate = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _write_audio(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sample_rate)


def source_audio_path(relative_wav: str, librispeech_root: Path) -> Path:
    """Resolve a LibriSpeechMix relative source path to existing WAV/FLAC audio."""
    rel = Path(relative_wav)
    wav_path = librispeech_root / rel
    flac_path = librispeech_root / rel.with_suffix(".flac")
    if wav_path.exists():
        return wav_path
    if flac_path.exists():
        return flac_path
    raise FileNotFoundError(f"Missing LibriSpeech source audio for {relative_wav} under {librispeech_root}")


def materialize_mixed_wav(row: dict, output_path: Path, librispeech_root: Path) -> Path:
    """Create a mixed WAV from source utterances and official LibriSpeechMix delays."""
    if output_path.exists():
        return output_path

    tracks = []
    sample_rate = None
    target_len = 0
    for relative_wav, delay in zip(row["wavs"], row["delays"]):
        audio, current_rate = _read_audio(source_audio_path(relative_wav, librispeech_root))
        if sample_rate is None:
            sample_rate = current_rate
        elif current_rate != sample_rate:
            raise RuntimeError(f"Sample-rate mismatch while mixing {row['id']}: {sample_rate} vs {current_rate}")
        delay_frames = int(round(float(delay) * current_rate))
        tracks.append((audio, delay_frames))
        target_len = max(target_len, delay_frames + len(audio))

    mixed = np.zeros(target_len, dtype=np.float32)
    for audio, delay_frames in tracks:
        mixed[delay_frames : delay_frames + len(audio)] += audio
    _write_audio(output_path, mixed, sample_rate or 16000)
    return output_path


def _public_audio_path(local_audio_path: Path, local_audio_root: Path, audio_prefix: Path) -> str:
    relative = local_audio_path.relative_to(local_audio_root)
    return str(audio_prefix / relative)


def build_record(
    row: dict,
    *,
    split: str,
    mix: str,
    data_dir: Path,
    audio_prefix: Path,
    no_audio: bool,
    librispeech_root: Path,
) -> dict:
    """Build one NeMo SOT manifest record and write its RTTM/SegLST sidecars."""
    duration = mixed_duration(row)
    session_id = Path(row["mixed_wav"]).stem
    local_audio_root = data_dir / "audio"
    local_audio_path = local_audio_root / row["mixed_wav"]
    rttm_path = data_dir / "rttm" / split / mix / f"{session_id}.rttm"
    seglst_path = data_dir / "seglst" / split / mix / f"{session_id}_ref.seglst.json"

    if not no_audio:
        materialize_mixed_wav(row, local_audio_path, librispeech_root)

    rttm_path.parent.mkdir(parents=True, exist_ok=True)
    rttm_path.write_text("".join(rttm_lines(row, session_id)), encoding="utf-8")
    seglst_path.parent.mkdir(parents=True, exist_ok=True)
    seglst_path.write_text(json.dumps(seglst_entries(row, session_id), indent=2) + "\n", encoding="utf-8")

    tag_by_speaker = speaker_tag_map(row)
    ordered_global_speakers = [speaker for speaker, _tag in sorted(tag_by_speaker.items(), key=lambda item: item[1])]
    audio_filepath = _public_audio_path(local_audio_path, local_audio_root, audio_prefix)
    text = sot_text(row)

    return {
        "audio_filepath": audio_filepath,
        "rttm_filepath": str(rttm_path),
        "reference_seglst_filepath": str(seglst_path),
        "offset": 0.0,
        "duration": round(duration, 4),
        "text": text,
        "expected_answer": text,
        "source_lang": "en",
        "target_lang": "en",
        "taskname": "asr",
        "pnc": "no",
        "num_speakers": len(tag_by_speaker),
        "num_changes": num_speaker_changes(row),
        "dataset_id": f"{BENCHMARK_NAME}:{split}-{mix}",
        "global_speaker_ids": ordered_global_speakers,
        "speaker_tag_map": tag_by_speaker,
        "librispeechmix_id": row["id"],
        "sample_id": session_id,
        "delays": row["delays"],
        "source_wavs": row["wavs"],
        "subset_for_metrics": f"{duration_bucket(duration)}-{split}-{mix}",
        "messages": [
            {
                "role": "user",
                "content": DEFAULT_PROMPT,
                "audio": {"path": audio_filepath, "duration": round(duration, 4)},
            }
        ],
    }


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    """Write JSONL rows and return the number written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "wt", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row) + "\n")
            count += 1
    return count


def prepare_split_mix(
    *,
    split: str,
    mix: str,
    rows: list[dict],
    output_dir: Path,
    data_dir: Path,
    audio_prefix: Path,
    no_audio: bool,
    max_samples: int | None,
    librispeech_root: Path,
) -> dict[str, int]:
    """Prepare under/over-20s variants for one LibriSpeechMix list."""
    buckets = {"under20s": [], "over20s": []}
    for row in rows:
        bucket = duration_bucket(mixed_duration(row))
        if max_samples is not None and len(buckets[bucket]) >= max_samples:
            continue
        buckets[bucket].append(
            build_record(
                row,
                split=split,
                mix=mix,
                data_dir=data_dir,
                audio_prefix=audio_prefix,
                no_audio=no_audio,
                librispeech_root=librispeech_root,
            )
        )
        if max_samples is not None and all(len(items) >= max_samples for items in buckets.values()):
            break

    counts = {}
    for bucket, records in buckets.items():
        out_file = output_dir / f"{bucket}-{split}-{mix}" / "test.jsonl"
        counts[f"{bucket}-{split}-{mix}"] = write_jsonl(out_file, records)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LibriSpeechMix SOT benchmark manifests.")
    parser.add_argument("--data_dir", type=str, default=None, help="External data/artifact root.")
    parser.add_argument("--audio-prefix", type=str, default=None, help="Audio path prefix embedded in manifests.")
    parser.add_argument("--no-audio", action="store_true", help="Do not download or materialize audio.")
    parser.add_argument("--splits", nargs="+", default=["test-clean"], help="LibriSpeechMix splits or 'all'.")
    parser.add_argument("--mixes", nargs="+", default=list(MIXES), help="Mixtures to prepare or 'all'.")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum records per under/over bucket.")
    parser.add_argument("--librispeechmix-dir", type=str, default=None, help="Local LibriSpeechMix checkout/list dir.")
    parser.add_argument("--librispeech-root", type=str, default=None, help="Existing LibriSpeech root override.")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else _default_data_dir()
    audio_prefix = Path(args.audio_prefix).expanduser() if args.audio_prefix else data_dir / "audio"
    lsm_dir = Path(args.librispeechmix_dir).expanduser().resolve() if args.librispeechmix_dir else None
    output_dir = Path(__file__).parent
    splits = _resolve_values(args.splits, SPLITS)
    mixes = _resolve_values(args.mixes, MIXES)

    data_dir.mkdir(parents=True, exist_ok=True)
    all_counts = {}
    for split in splits:
        librispeech_root = (
            Path(args.librispeech_root).expanduser().resolve()
            if args.librispeech_root
            else ensure_librispeech_audio(split, data_dir, args.no_audio)
        )
        for mix in mixes:
            name = f"{split}-{mix}"
            list_path = ensure_librispeechmix_list(name, data_dir, lsm_dir)
            rows = load_jsonl(list_path)
            counts = prepare_split_mix(
                split=split,
                mix=mix,
                rows=rows,
                output_dir=output_dir,
                data_dir=data_dir,
                audio_prefix=audio_prefix,
                no_audio=args.no_audio,
                max_samples=args.max_samples,
                librispeech_root=librispeech_root,
            )
            all_counts.update(counts)

    print("Prepared LibriSpeechMix SOT manifests:")
    for name, count in sorted(all_counts.items()):
        print(f"  {name}: {count}")
    print(f"Data/artifacts root: {data_dir}")
    print(f"Duration split threshold: {DURATION_THRESHOLD_SECONDS:.1f}s")


if __name__ == "__main__":
    main()
