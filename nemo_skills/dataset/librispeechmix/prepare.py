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

"""Prepare LibriSpeechMix for NeMo Skills evaluation."""

import argparse
import gzip
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

BENCHMARK_NAME = "librispeechmix"
SYSTEM_MESSAGE = "You are a helpful assistant. /no_think"
ASR_PROMPT = (
    "Transcribe every speaker in the mixed audio. Return one utterance per line in any order, "
    "with no speaker labels."
)
SA_ASR_PROFILE_PROMPT = "These audio clips are enrollment samples for speaker_{profile_index}."
SA_ASR_PROMPT = (
    "Transcribe the mixed audio using the speaker references above. Return one line per speaker in any order "
    "using the format 'speaker_<profile_index>: <transcript>'."
)

SPLITS = ("dev-clean", "test-clean")
MIXES = ("1mix", "2mix", "3mix")
MODES = ("asr", "sa-asr")

OPENSLR_ARCHIVES = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
}

UPSTREAM_MANIFESTS = {
    "dev-clean-1mix": {
        "rows": 2703,
        "sha256": "6cd4a30e8aa1fdb41a077ae57e67f5cedbf2d552097cea2b63ecf1522cd7448e",
    },
    "dev-clean-2mix": {
        "rows": 2703,
        "sha256": "27137e7a3293ddbb97f12584449e25d025f7581205e98b7c59787bb223c98abc",
    },
    "dev-clean-3mix": {
        "rows": 2703,
        "sha256": "06f208aeefb7b363c49fafcbbc13374063e338a4bd0ae4fc6598c022b80e2399",
    },
    "test-clean-1mix": {
        "rows": 2620,
        "sha256": "7c5a40322f7e578cc6e47eb8d68cfd6b7cb20944a980f2a6afdc7ac7db9b1076",
    },
    "test-clean-2mix": {
        "rows": 2620,
        "sha256": "95a01945c855c1ec7d1dc6105beff70e6d35e3372a5e2bec40bcaa4b372d216d",
    },
    "test-clean-3mix": {
        "rows": 2620,
        "sha256": "b98134a8c5e5739c53d12a0e920242ee380c3be00dce6e5b77331d7d26db695e",
    },
}


def _absolute_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    return path if path.is_absolute() else path.resolve()


def _default_data_dir() -> Path:
    """Choose a dataset root outside the repo tree by default."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent.parent / f"{parent.name}-data"
    return Path.home() / ".cache" / "nemo-skills-data"


def _manifest_asset_path(manifest_name: str) -> Path:
    return Path(__file__).parent / "manifests" / f"{manifest_name}.jsonl.gz"


def _download_with_progress(url: str, output_path: Path, description: str) -> None:
    with tqdm(unit="B", unit_scale=True, unit_divisor=1024, desc=description) as progress_bar:

        def report_hook(block_num: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                progress_bar.total = total_size
            progress_bar.update(max(0, block_num * block_size - progress_bar.n))

        urllib.request.urlretrieve(url, output_path, report_hook)


def _safe_extract_archive(archive_path: Path, output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    with tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (output_dir / member.name).resolve()
            if output_dir not in member_path.parents and member_path != output_dir:
                raise RuntimeError(f"Refusing to extract path outside target directory: {member.name}")
        tar.extractall(output_dir)


def ensure_openslr_split(split_name: str, raw_root: Path) -> None:
    """Download and extract LibriSpeech dev/test-clean if needed."""
    split_dir = raw_root / "LibriSpeech" / split_name
    if split_dir.exists():
        return

    raw_root.mkdir(parents=True, exist_ok=True)
    archive_path = raw_root / f"{split_name}.tar.gz"

    print("=" * 72)
    print(f"DOWNLOADING LibriSpeech {split_name}")
    print(f"Destination: {raw_root}")
    print("WARNING: This pulls the official OpenSLR archive and may take several minutes.")
    print("=" * 72)

    _download_with_progress(OPENSLR_ARCHIVES[split_name], archive_path, f"Downloading {split_name}")
    _safe_extract_archive(archive_path, raw_root)
    archive_path.unlink()


def load_upstream_manifest(manifest_name: str) -> list[dict]:
    """Load a vendored official LibriSpeechMix manifest and verify fidelity."""
    manifest_path = _manifest_asset_path(manifest_name)
    payload = gzip.decompress(manifest_path.read_bytes())
    metadata = UPSTREAM_MANIFESTS[manifest_name]
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    if payload_sha256 != metadata["sha256"]:
        raise RuntimeError(
            f"Vendored manifest {manifest_name} failed SHA256 validation: "
            f"{payload_sha256} != {metadata['sha256']}"
        )

    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    if len(rows) != metadata["rows"]:
        raise RuntimeError(
            f"Vendored manifest {manifest_name} has {len(rows)} rows, expected {metadata['rows']}."
        )
    return rows


def ensure_source_wav(relative_wav_path: str, source_audio_root: Path, raw_librispeech_root: Path) -> tuple[Path, float]:
    """Convert a source FLAC utterance to a WAV cache entry and return its path and duration."""
    output_path = source_audio_root / relative_wav_path
    if output_path.exists():
        info = sf.info(output_path)
        return output_path.resolve(), info.frames / info.samplerate

    input_path = raw_librispeech_root / Path(relative_wav_path).with_suffix(".flac")
    if not input_path.exists():
        raise FileNotFoundError(f"Missing LibriSpeech source audio: {input_path}")

    audio, sample_rate = sf.read(input_path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sample_rate)
    return output_path.resolve(), len(audio) / sample_rate


def mix_audio_sources(
    relative_wav_paths: list[str],
    delays: list[float],
    mixed_output_path: Path,
    source_audio_root: Path,
    raw_librispeech_root: Path,
) -> tuple[Path, float]:
    """Create a mixed LibriSpeechMix WAV using the official manifest delays."""
    if mixed_output_path.exists():
        info = sf.info(mixed_output_path)
        return mixed_output_path.resolve(), info.frames / info.samplerate

    tracks: list[tuple[np.ndarray, int]] = []
    sample_rate = None
    target_length = 0

    for relative_wav_path, delay in zip(relative_wav_paths, delays):
        source_path, _ = ensure_source_wav(relative_wav_path, source_audio_root, raw_librispeech_root)
        audio, current_sample_rate = sf.read(source_path)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        audio = np.asarray(audio, dtype=np.float32)

        if sample_rate is None:
            sample_rate = current_sample_rate
        elif sample_rate != current_sample_rate:
            raise RuntimeError(f"Mismatched sample rates in LibriSpeechMix sources: {sample_rate} vs {current_sample_rate}")

        delay_frames = int(round(delay * current_sample_rate))
        target_length = max(target_length, delay_frames + len(audio))
        tracks.append((audio, delay_frames))

    mixed_audio = np.zeros(target_length, dtype=np.float32)
    for audio, delay_frames in tracks:
        mixed_audio[delay_frames : delay_frames + len(audio)] += audio

    mixed_output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(mixed_output_path, mixed_audio, sample_rate)
    return mixed_output_path.resolve(), len(mixed_audio) / sample_rate


def _estimate_mixed_duration(row: dict) -> float:
    return max(delay + duration for delay, duration in zip(row["delays"], row["durations"]))


def _system_message() -> dict:
    return {"role": "system", "content": SYSTEM_MESSAGE}


def _asr_messages(audio_path: str, duration: float | None) -> list[dict]:
    audio_metadata = {"path": audio_path}
    if duration is not None:
        audio_metadata["duration"] = float(duration)
    return [_system_message(), {"role": "user", "content": ASR_PROMPT, "audio": audio_metadata}]


def _sa_asr_messages(
    mixed_audio_path: str, mixed_duration: float | None, speaker_profile_audio: list[list[dict]]
) -> list[dict]:
    messages = [_system_message()]
    for profile_index, audio_group in enumerate(speaker_profile_audio):
        messages.append(
            {
                "role": "user",
                "content": SA_ASR_PROFILE_PROMPT.format(profile_index=profile_index),
                "audios": audio_group,
            }
        )

    mixed_audio_metadata = {"path": mixed_audio_path}
    if mixed_duration is not None:
        mixed_audio_metadata["duration"] = float(mixed_duration)
    messages.append({"role": "user", "content": SA_ASR_PROMPT, "audio": mixed_audio_metadata})
    return messages


def build_output_record(
    row: dict,
    mode: str,
    mixed_audio_path: Path,
    mixed_duration: float | None,
    audio_prefix: Path,
    source_audio_root: Path,
    raw_librispeech_root: Path,
    materialize_audio: bool,
) -> dict:
    """Build one NeMo Skills-ready JSONL entry."""
    public_audio_root = _absolute_path(audio_prefix)
    public_source_root = public_audio_root / "source"
    public_mixed_root = public_audio_root / "mixed"

    if materialize_audio:
        _, mixed_duration = mix_audio_sources(
            row["wavs"],
            row["delays"],
            mixed_audio_path,
            source_audio_root,
            raw_librispeech_root,
        )
    elif mixed_duration is None:
        mixed_duration = _estimate_mixed_duration(row)

    source_audio_public_paths = []
    speaker_profile_audio = []
    for audio_group in row["speaker_profile"]:
        public_group = []
        for relative_wav_path in audio_group:
            public_path = public_source_root / relative_wav_path
            duration = None
            if materialize_audio:
                _, duration = ensure_source_wav(relative_wav_path, source_audio_root, raw_librispeech_root)
            audio_metadata = {"path": str(public_path)}
            if duration is not None:
                audio_metadata["duration"] = float(duration)
            public_group.append(audio_metadata)
        speaker_profile_audio.append(public_group)

    for relative_wav_path in row["wavs"]:
        public_source_path = public_source_root / relative_wav_path
        if materialize_audio:
            ensure_source_wav(relative_wav_path, source_audio_root, raw_librispeech_root)
        source_audio_public_paths.append(str(public_source_path))

    mixed_audio_public_path = str(public_mixed_root / row["mixed_wav"])
    manifest_name = Path(row["mixed_wav"]).parts[0]

    if mode == "asr":
        messages = _asr_messages(mixed_audio_public_path, mixed_duration)
        expected_answer = "\n".join(row["texts"])
        task_type = "LIBRISPEECHMIX_ASR"
    else:
        labeled_references = [
            f"speaker_{speaker_idx}: {text}"
            for speaker_idx, text in sorted(zip(row["speaker_profile_index"], row["texts"]), key=lambda item: item[0])
        ]
        messages = _sa_asr_messages(mixed_audio_public_path, mixed_duration, speaker_profile_audio)
        expected_answer = "\n".join(labeled_references)
        task_type = "LIBRISPEECHMIX_SA_ASR"

    return {
        "id": row["id"],
        "task_type": task_type,
        "expected_answer": expected_answer,
        "reference_streams": row["texts"],
        "speaker_profile_index": row["speaker_profile_index"],
        "messages": messages,
        "subset_for_metrics": f"{mode}-{manifest_name}",
        "audio_duration": float(mixed_duration) if mixed_duration is not None else None,
        "mixed_audio_path": mixed_audio_public_path,
        "speaker_profile": [[audio["path"] for audio in group] for group in speaker_profile_audio],
        "source_wavs": source_audio_public_paths,
        "wavs": row["wavs"],
        "delays": row["delays"],
        "durations": row["durations"],
        "speakers": row["speakers"],
        "genders": row["genders"],
        "num_speakers": len(row["texts"]),
    }


def write_benchmark_records(output_file: Path, records: list[dict]) -> None:
    """Write benchmark records to JSONL."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_benchmark(
    manifest_name: str,
    mode: str,
    dataset_root: Path,
    audio_prefix: Path,
    materialize_audio: bool,
    max_samples: int | None = None,
) -> Path:
    """Prepare one LibriSpeechMix benchmark variant."""
    split_name = manifest_name.rsplit("-", 1)[0]
    raw_root = dataset_root / "raw"
    raw_librispeech_root = raw_root / "LibriSpeech"
    source_audio_root = dataset_root / "audio" / "source"
    mixed_audio_root = dataset_root / "audio" / "mixed"

    if materialize_audio:
        ensure_openslr_split(split_name, raw_root)

    rows = load_upstream_manifest(manifest_name)
    if max_samples is not None:
        rows = rows[:max_samples]

    output_file = dataset_root / f"{mode}-{manifest_name}" / "test.jsonl"
    records = []
    for row in tqdm(rows, desc=f"{mode}-{manifest_name}"):
        mixed_audio_path = mixed_audio_root / row["mixed_wav"]
        records.append(
            build_output_record(
                row=row,
                mode=mode,
                mixed_audio_path=mixed_audio_path,
                mixed_duration=None,
                audio_prefix=audio_prefix,
                source_audio_root=source_audio_root,
                raw_librispeech_root=raw_librispeech_root,
                materialize_audio=materialize_audio,
            )
        )

    write_benchmark_records(output_file, records)
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare LibriSpeechMix for NeMo Skills")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=(
            "Base output directory. Prepared files are written under <data_dir>/librispeechmix/. "
            "When omitted, defaults to a sibling directory of the repo so raw audio stays outside the tree."
        ),
    )
    parser.add_argument(
        "--audio-prefix",
        type=str,
        default=None,
        help=(
            "Absolute prefix written into JSONL audio paths. Defaults to <data_dir>/librispeechmix/audio. "
            "Use this when evaluation sees the prepared audio through a different mount point."
        ),
    )
    parser.add_argument("--no-audio", action="store_true", help="Write manifests without downloading or materializing audio.")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS), help="Subset of splits to prepare.")
    parser.add_argument("--mixes", nargs="+", choices=MIXES, default=list(MIXES), help="Subset of mixes to prepare.")
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES), help="Subset of evaluation modes to prepare.")
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on rows per manifest for smoke testing. Defaults to all rows.",
    )
    args = parser.parse_args()

    base_data_dir = _absolute_path(args.data_dir) if args.data_dir else _default_data_dir()
    dataset_root = base_data_dir / BENCHMARK_NAME
    audio_prefix = _absolute_path(args.audio_prefix) if args.audio_prefix else dataset_root / "audio"

    materialize_audio = not args.no_audio

    print(f"Dataset root: {dataset_root}")
    print(f"Audio prefix in JSONL: {audio_prefix}")
    if materialize_audio:
        print("Audio materialization: enabled")
    else:
        print("Audio materialization: disabled (--no-audio)")

    prepared_outputs = []
    for split_name in args.splits:
        for mix_name in args.mixes:
            manifest_name = f"{split_name}-{mix_name}"
            for mode in args.modes:
                prepared_outputs.append(
                    prepare_benchmark(
                        manifest_name=manifest_name,
                        mode=mode,
                        dataset_root=dataset_root,
                        audio_prefix=audio_prefix,
                        materialize_audio=materialize_audio,
                        max_samples=args.max_samples,
                    )
                )

    print("\nPrepared manifests:")
    for output_path in prepared_outputs:
        print(f"  {output_path}")


if __name__ == "__main__":
    main()
