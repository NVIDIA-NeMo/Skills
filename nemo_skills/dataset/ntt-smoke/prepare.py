# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Prepare NTT-SMOKE manifests from already-prepared NeMo-Skills data.

NTT-SMOKE intentionally reuses small, representative slices from existing
prepared benchmarks. Source samples keep origin metadata, while generated noisy
and long-form samples are written under ``ntt-smoke/data``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

SYSTEM_MESSAGE = "You are a helpful assistant. /no_think"
DEFAULT_SOURCE_DATA_DIR = (
    "/home/vmendelev/.cache/saferun/cluster-sshfs/oci_iad/lustre/fs12/portfolios/llmservice/users/"
    "pzelasko/results/speechlm-2026h1/skills_data"
)
DEFAULT_PREFERENCE_ASR_DIR = (
    "/lustre/fsw/portfolios/llmservice/projects/llmservice_nemo_speechlm/data/ASR/preference-asr-bench"
)
DEFAULT_LANGUAGES = [
    "ar_eg",
    "bg_bg",
    "cmn_hans_cn",
    "hr_hr",
    "cs_cz",
    "da_dk",
    "nl_nl",
    "en_us",
    "et_ee",
    "fi_fi",
    "fr_fr",
    "de_de",
    "el_gr",
    "he_il",
    "hi_in",
    "hu_hu",
    "it_it",
    "ja_jp",
    "ko_kr",
    "lv_lv",
    "lt_lt",
    "mt_mt",
    "pl_pl",
    "pt_br",
    "ro_ro",
    "ru_ru",
    "sk_sk",
    "sl_si",
    "es_419",
    "sv_se",
    "th_th",
    "uk_ua",
]

COVOST2_BY_FLEURS = {
    "ar_eg": "ar",
    "cmn_hans_cn": "zh-CN",
    "nl_nl": "nl",
    "en_us": "en",
    "et_ee": "et",
    "fr_fr": "fr",
    "de_de": "de",
    "it_it": "it",
    "ja_jp": "ja",
    "lv_lv": "lv",
    "pt_br": "pt",
    "ru_ru": "ru",
    "sl_si": "sl",
    "es_419": "es",
    "sv_se": "sv-SE",
}

ASR_PROMPTS = {
    "canonical": "Transcribe the following audio.",
    "terse": "Transcribe this audio.",
    "verbatim": "Please transcribe the speech in the audio verbatim.",
    "plain": "Write exactly what is spoken in the audio.",
}

NOISE_SNRS = [5.0, 10.0, 15.0]
LONG_TARGET_SECONDS = [20 * 60, 40 * 60, 60 * 60]
SOURCE_READ_LIMIT = 0


def _stable_key(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as fin:
        for idx, line in enumerate(fin):
            if SOURCE_READ_LIMIT > 0 and idx >= SOURCE_READ_LIMIT:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _sample(rows: list[dict[str, Any]], count: int, salt: str) -> list[dict[str, Any]]:
    if count <= 0 or not rows:
        return []
    keyed = sorted(
        rows,
        key=lambda row: _stable_key(
            [salt, row.get("id"), row.get("sample_id"), row] if isinstance(row, dict) else [salt, row]
        ),
    )
    return keyed[: min(count, len(keyed))]


def _duration(row: dict[str, Any]) -> float | None:
    duration = row.get("duration") or row.get("audio_duration")
    if duration is not None:
        return float(duration)
    for message in row.get("messages") or []:
        audio = message.get("audio") or {}
        if audio.get("duration") is not None:
            return float(audio["duration"])
    return None


def _audio_path(row: dict[str, Any]) -> str | None:
    for key in ("audio_filepath", "audio_path"):
        value = row.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, list) and value:
            return value[0]
    for message in row.get("messages") or []:
        audio = message.get("audio") or {}
        if audio.get("path"):
            return audio["path"]
    return None


def _resolve_audio_path(audio_path: str, source_root: Path) -> Path:
    path = Path(audio_path)
    if path.exists():
        return path
    for prefix in ("/data/", "/dataset/"):
        if audio_path.startswith(prefix):
            return source_root / audio_path[len(prefix) :]
    return path


def _set_user_prompt(row: dict[str, Any], prompt: str) -> None:
    messages = row.setdefault("messages", [])
    user_message = None
    for message in messages:
        if message.get("role") == "user":
            user_message = message
            break
    if user_message is None:
        user_message = {"role": "user"}
        messages.append(user_message)
    user_message["content"] = prompt


def _ensure_system_message(row: dict[str, Any]) -> None:
    messages = row.setdefault("messages", [])
    if not any(message.get("role") == "system" for message in messages):
        messages.insert(0, {"role": "system", "content": SYSTEM_MESSAGE})


def _origin_id(row: dict[str, Any]) -> str:
    for key in ("id", "sample_id", "uniq_id", "key"):
        if row.get(key) is not None:
            return str(row[key])
    audio_path = _audio_path(row)
    if audio_path:
        return Path(audio_path).stem
    return _stable_key(row)[:16]


def _with_metadata(
    row: dict[str, Any],
    *,
    variant: str,
    subtask: str,
    origin_dataset: str,
    origin_split: str,
    origin_manifest: str,
    language: str = "en",
    prompt_variant: str = "canonical",
    prompt_group_id: str | None = None,
    task_type: str | None = None,
    prompt: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = copy.deepcopy(row)
    if prompt is not None:
        _set_user_prompt(out, prompt)
    _ensure_system_message(out)
    if task_type is not None:
        out["task_type"] = task_type
    out["subset_for_metrics"] = subtask
    out["ntt_variant"] = variant
    out["ntt_subtask"] = subtask
    out["origin_dataset"] = origin_dataset
    out["origin_split"] = origin_split
    out["origin_manifest"] = origin_manifest
    out["origin_id"] = _origin_id(row)
    out["language"] = language
    out["prompt_variant"] = prompt_variant
    if prompt_group_id is not None:
        out["prompt_group_id"] = prompt_group_id
    if extra:
        out.update(extra)
    return out


def _source_rel(dataset: str, filename: str) -> str:
    return f"{dataset}/{filename}"


def _load_source(source_root: Path, dataset: str, filename: str) -> list[dict[str, Any]]:
    return _read_jsonl(source_root / _source_rel(dataset, filename))


def _rows_by_language(rows: list[dict[str, Any]], languages: list[str]) -> dict[str, list[dict[str, Any]]]:
    wanted = set(languages)
    wanted.update(COVOST2_BY_FLEURS.get(language, language) for language in languages)
    grouped: dict[str, list[dict[str, Any]]] = {language: [] for language in languages}
    for row in rows:
        extra_fields = row.get("extra_fields") or {}
        language = extra_fields.get("src_lang") or row.get("language")
        if language not in wanted:
            continue
        group_key = language
        for fleurs_locale, covost2_lang in COVOST2_BY_FLEURS.items():
            if language == covost2_lang and fleurs_locale in grouped:
                group_key = fleurs_locale
                break
        if group_key in grouped:
            grouped[group_key].append(row)
    return grouped


def _sample_by_language(rows: list[dict[str, Any]], count: int, languages: list[str], salt: str) -> list[dict[str, Any]]:
    grouped = _rows_by_language(rows, languages)
    sampled = []
    per_language = max(1, count // max(1, len(languages)))
    for language in languages:
        sampled.extend(_sample(grouped.get(language, []), per_language, f"{salt}:{language}"))
    return _sample(sampled, count, f"{salt}:trim")


def _read_audio(row: dict[str, Any], source_root: Path) -> tuple[np.ndarray, int] | None:
    audio_path = _audio_path(row)
    if not audio_path:
        return None
    local_path = _resolve_audio_path(audio_path, source_root)
    if not local_path.exists():
        return None
    audio, sample_rate = sf.read(str(local_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, int(sample_rate)


def _fit_noise(noise: np.ndarray, target_len: int, offset: int) -> np.ndarray:
    if len(noise) == 0:
        return np.zeros(target_len, dtype=np.float32)
    if len(noise) < target_len:
        repeats = int(np.ceil(target_len / len(noise)))
        noise = np.tile(noise, repeats)
    if len(noise) == target_len:
        return noise.astype(np.float32)
    start = offset % max(1, len(noise) - target_len)
    return noise[start : start + target_len].astype(np.float32)


def _mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    speech_power = float(np.mean(np.square(speech))) + 1e-12
    noise_power = float(np.mean(np.square(noise))) + 1e-12
    target_noise_power = speech_power / (10 ** (snr_db / 10.0))
    scaled_noise = noise * np.sqrt(target_noise_power / noise_power)
    mixed = speech + scaled_noise
    peak = float(np.max(np.abs(mixed))) if len(mixed) else 0.0
    if peak > 0.99:
        mixed = mixed * (0.99 / peak)
    return mixed.astype(np.float32)


def _create_noisy_rows(
    rows: list[dict[str, Any]],
    noise_rows: list[dict[str, Any]],
    *,
    source_root: Path,
    output_dir: Path,
    variant: str,
    subtask: str,
    origin_dataset: str,
    origin_split: str,
    origin_manifest: str,
    count: int,
    salt: str,
) -> list[dict[str, Any]]:
    selected = _sample(rows, count, salt)
    noise_selected = _sample(noise_rows, max(count, 1), f"{salt}:noise")
    generated = []
    for idx, row in enumerate(selected):
        speech = _read_audio(row, source_root)
        noise = _read_audio(noise_selected[idx % len(noise_selected)], source_root) if noise_selected else None
        if speech is None or noise is None:
            continue
        speech_audio, speech_sr = speech
        noise_audio, noise_sr = noise
        if speech_sr != noise_sr:
            continue
        snr = NOISE_SNRS[idx % len(NOISE_SNRS)]
        fitted_noise = _fit_noise(noise_audio, len(speech_audio), idx * 7919)
        mixed = _mix_at_snr(speech_audio, fitted_noise, snr)
        audio_rel = Path("data") / "noisy" / variant / f"{subtask.replace('.', '_')}_{idx:05d}.wav"
        audio_path = output_dir / audio_rel
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(audio_path), mixed, speech_sr)
        container_path = f"/data/ntt-smoke/{audio_rel.as_posix()}"

        noisy_row = _with_metadata(
            row,
            variant=variant,
            subtask=subtask,
            origin_dataset=origin_dataset,
            origin_split=origin_split,
            origin_manifest=origin_manifest,
            extra={
                "audio_filepath": container_path,
                "audio_path": container_path,
                "noise_snr_db": snr,
                "noise_origin_id": _origin_id(noise_selected[idx % len(noise_selected)]),
            },
        )
        for message in noisy_row.get("messages") or []:
            audio_meta = message.get("audio")
            if isinstance(audio_meta, dict):
                audio_meta["path"] = container_path
                if _duration(row) is not None:
                    audio_meta["duration"] = _duration(row)
        generated.append(noisy_row)
    return generated


def _create_long_rows(
    rows: list[dict[str, Any]],
    *,
    source_root: Path,
    output_dir: Path,
    variant: str,
    count: int,
    salt: str,
) -> list[dict[str, Any]]:
    source_rows = _sample([row for row in rows if _duration(row)], max(len(rows), count * 20), salt)
    if not source_rows:
        return []
    long_rows = []
    cursor = 0
    for idx in range(count):
        target_seconds = LONG_TARGET_SECONDS[idx % len(LONG_TARGET_SECONDS)]
        chunks = []
        texts = []
        origins = []
        sample_rate = None
        total = 0.0
        attempts = 0
        while total < target_seconds and attempts < len(source_rows) * 3:
            row = source_rows[cursor % len(source_rows)]
            cursor += 1
            attempts += 1
            audio = _read_audio(row, source_root)
            if audio is None:
                continue
            samples, sr = audio
            if sample_rate is None:
                sample_rate = sr
            if sr != sample_rate:
                continue
            chunks.append(samples)
            chunks.append(np.zeros(int(0.5 * sr), dtype=np.float32))
            texts.append(str(row.get("expected_answer", "")).strip())
            origins.append(_origin_id(row))
            total += len(samples) / sr + 0.5
        if not chunks or sample_rate is None:
            continue
        stitched = np.concatenate(chunks)
        audio_rel = Path("data") / "long" / variant / f"ntt_smoke_long_{idx:03d}.wav"
        audio_path = output_dir / audio_rel
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(audio_path), stitched, sample_rate)
        container_path = f"/data/ntt-smoke/{audio_rel.as_posix()}"
        duration = len(stitched) / sample_rate
        expected_answer = " ".join(text for text in texts if text)
        row = {
            "task_type": "ASR",
            "expected_answer": expected_answer,
            "audio_filepath": container_path,
            "audio_path": container_path,
            "audio_duration": duration,
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": ASR_PROMPTS["canonical"],
                    "audio": {"path": container_path, "duration": duration},
                },
            ],
            "stitched_origin_ids": origins,
        }
        long_rows.append(
            _with_metadata(
                row,
                variant=variant,
                subtask="asr.long",
                origin_dataset="asr-leaderboard",
                origin_split="stitched",
                origin_manifest="asr-leaderboard/earnings22.jsonl+tedlium.jsonl",
                extra={"stitched_target_seconds": target_seconds},
            )
        )
    return long_rows


def _context_rows(source_root: Path, variant: str, samples_per_mode: int) -> list[dict[str, Any]]:
    modes = ["contextless", "coarse", "fine"]
    by_mode = {
        mode: _load_source(source_root, "contextasr-bench", f"{mode}/test.jsonl")
        for mode in modes
    }
    common_ids = set.intersection(*[set(row.get("uniq_id") for row in rows) for rows in by_mode.values() if rows])
    if not common_ids:
        return []
    selected_ids = {
        row.get("uniq_id")
        for row in _sample([row for row in by_mode["fine"] if row.get("uniq_id") in common_ids], samples_per_mode, variant)
    }
    out = []
    for mode in modes:
        mode_rows = [row for row in by_mode[mode] if row.get("uniq_id") in selected_ids]
        for row in _sample(mode_rows, samples_per_mode, f"{variant}:{mode}"):
            out.append(
                _with_metadata(
                    row,
                    variant=variant,
                    subtask=f"context_biasing.{mode}",
                    origin_dataset="contextasr-bench",
                    origin_split=mode,
                    origin_manifest=f"contextasr-bench/{mode}/test.jsonl",
                    task_type="ContextASR",
                    prompt_group_id=f"context:{row.get('uniq_id')}",
                    extra={"context_mode": mode},
                )
            )
    return out


def _text_rows(source_root: Path, variant: str, count: int) -> list[dict[str, Any]]:
    gpqa = _load_source(source_root, "gpqa", "diamond.jsonl")
    out = []
    for row in _sample(gpqa, count, f"{variant}:text"):
        prompt = row.get("problem") or row.get("question")
        if not prompt:
            continue
        text_row = {
            "task_type": "Text-MCQ",
            "expected_answer": row.get("expected_answer"),
            "messages": [{"role": "user", "content": prompt}],
            "problem": prompt,
        }
        out.append(
            _with_metadata(
                text_row,
                variant=variant,
                subtask="text.superficial",
                origin_dataset="gpqa",
                origin_split="diamond",
                origin_manifest="gpqa/diamond.jsonl",
                language="en",
            )
        )
    return out


def _iter_preference_asr_rows(preference_dir: Path, max_rows: int) -> Iterable[tuple[Path, dict[str, Any]]]:
    if not preference_dir.exists():
        return
    seen = 0
    for manifest in sorted(preference_dir.rglob("*.jsonl")):
        with open(manifest, encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if _audio_path(row) is None and not row.get("audio_filepath"):
                    continue
                expected = row.get("expected_answer") or row.get("text") or row.get("transcript") or row.get("answer")
                if not expected:
                    continue
                seen += 1
                yield manifest, row
                if max_rows > 0 and seen >= max_rows:
                    return


def _preference_audio_instruction_rows(preference_dir: Path, variant: str, count: int) -> list[dict[str, Any]]:
    if not preference_dir.exists():
        return []
    candidates = list(_iter_preference_asr_rows(preference_dir, max(count * 20, 200)))
    out = []
    for manifest, row in _sample(candidates, count, f"{variant}:preference_asr"):
        source_row = copy.deepcopy(row)
        expected = (
            source_row.get("expected_answer")
            or source_row.get("text")
            or source_row.get("transcript")
            or source_row.get("answer")
        )
        source_row["expected_answer"] = expected
        source_row.setdefault("task_type", "ASR-PC")
        audio_path = _audio_path(source_row) or source_row.get("audio_filepath")
        if not source_row.get("messages"):
            source_row["messages"] = [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {
                    "role": "user",
                    "content": "Transcribe the audio and preserve the requested punctuation, capitalization, and formatting.",
                    "audio": {"path": audio_path},
                },
            ]
        out.append(
            _with_metadata(
                source_row,
                variant=variant,
                subtask="audio_instruction_following.preference_asr",
                origin_dataset="preference-asr-bench",
                origin_split=manifest.parent.name,
                origin_manifest=str(manifest),
                prompt=(
                    "Transcribe the audio and preserve the requested punctuation, capitalization, and formatting."
                ),
                prompt_variant="preference_asr",
                task_type=source_row.get("task_type") or "ASR-PC",
            )
        )
    return out


def _hallucination_rows(source_root: Path, variant: str, count: int) -> list[dict[str, Any]]:
    musan = _load_source(source_root, "musan", "test.jsonl")
    out = []
    for idx, row in enumerate(_sample(musan, count, f"{variant}:musan")):
        if idx % 2 == 0:
            prompt = ASR_PROMPTS["canonical"]
            prompt_variant = "production_asr"
        else:
            prompt = "Transcribe the speech in this audio. If there is no speech, do not output anything."
            prompt_variant = "explicit_abstention"
        out.append(
            _with_metadata(
                row,
                variant=variant,
                subtask="hallucination.nonspeech",
                origin_dataset="musan",
                origin_split=row.get("category", "test"),
                origin_manifest="musan/test.jsonl",
                prompt=prompt,
                prompt_variant=prompt_variant,
            )
        )
    return out


def _prompt_robustness_rows(rows: list[dict[str, Any]], variant: str, base_count: int, salt: str) -> list[dict[str, Any]]:
    out = []
    for row in _sample(rows, base_count, salt):
        group_id = f"prompt:{variant}:{_origin_id(row)}"
        for prompt_variant, prompt in ASR_PROMPTS.items():
            out.append(
                _with_metadata(
                    row,
                    variant=variant,
                    subtask="prompt_robustness",
                    origin_dataset=row.get("origin_dataset", "asr-leaderboard"),
                    origin_split=row.get("origin_split", row.get("subset_for_metrics", "test")),
                    origin_manifest=row.get("origin_manifest", "asr-leaderboard/test.jsonl"),
                    prompt=prompt,
                    prompt_variant=prompt_variant,
                    prompt_group_id=group_id,
                    language=row.get("language", "en"),
                )
            )
    return out


def _audio_instruction_rows(
    source_root: Path,
    preference_dir: Path,
    variant: str,
    count: int,
) -> list[dict[str, Any]]:
    preference_rows = _preference_audio_instruction_rows(preference_dir, variant, count)
    if preference_rows:
        return preference_rows

    pc_rows = _load_source(source_root, "librispeech-pc", "test-clean.jsonl")
    out = []
    for row in _sample(pc_rows, count, f"{variant}:pc"):
        out.append(
            _with_metadata(
                row,
                variant=variant,
                subtask="audio_instruction_following.punctuation_capitalization",
                origin_dataset="librispeech-pc",
                origin_split=row.get("split", "test-clean"),
                origin_manifest="librispeech-pc/test-clean.jsonl",
                prompt="Transcribe the audio with proper punctuation and capitalization.",
                prompt_variant="punctuation_capitalization",
            )
        )
    return out


def _english_rows(source_root: Path, output_dir: Path, args: argparse.Namespace, variant: str = "en") -> list[dict[str, Any]]:
    n = args.audio_samples
    clean_read = _load_source(source_root, "asr-leaderboard", "librispeech_clean.jsonl")
    clean_conv = _load_source(source_root, "asr-leaderboard", "ami.jsonl")
    clean_media = _load_source(source_root, "asr-leaderboard", "tedlium.jsonl") + _load_source(
        source_root, "asr-leaderboard", "gigaspeech.jsonl"
    )
    media_for_noise = clean_media + _load_source(source_root, "asr-leaderboard", "voxpopuli.jsonl")
    short_pool = [
        row
        for row in clean_read + clean_conv + clean_media + _load_source(source_root, "asr-leaderboard", "librispeech_other.jsonl")
        if (_duration(row) or 999.0) <= args.short_max_seconds
    ]
    long_pool = _load_source(source_root, "asr-leaderboard", "earnings22.jsonl") + _load_source(
        source_root, "asr-leaderboard", "tedlium.jsonl"
    )
    noise_rows = _load_source(source_root, "musan", "test.jsonl")

    rows: list[dict[str, Any]] = []
    for subtask, source_rows, manifest, count in [
        ("asr.clean_read", clean_read, "asr-leaderboard/librispeech_clean.jsonl", n // 3),
        ("asr.clean_conversational", clean_conv, "asr-leaderboard/ami.jsonl", n // 3),
        ("asr.clean_media", clean_media, "asr-leaderboard/tedlium.jsonl+gigaspeech.jsonl", n - 2 * (n // 3)),
        ("asr.short", short_pool, "asr-leaderboard/*", n),
    ]:
        for row in _sample(source_rows, count, f"{variant}:{subtask}"):
            rows.append(
                _with_metadata(
                    row,
                    variant=variant,
                    subtask=subtask,
                    origin_dataset="asr-leaderboard",
                    origin_split=row.get("subset_for_metrics", "test"),
                    origin_manifest=manifest,
                    language="en",
                )
            )

    rows.extend(
        _create_noisy_rows(
            clean_conv,
            noise_rows,
            source_root=source_root,
            output_dir=output_dir,
            variant=variant,
            subtask="asr.noisy_conversational",
            origin_dataset="asr-leaderboard",
            origin_split="ami",
            origin_manifest="asr-leaderboard/ami.jsonl",
            count=n,
            salt=f"{variant}:noisy_conv",
        )
    )
    rows.extend(
        _create_noisy_rows(
            media_for_noise,
            noise_rows,
            source_root=source_root,
            output_dir=output_dir,
            variant=variant,
            subtask="asr.noisy_media",
            origin_dataset="asr-leaderboard",
            origin_split="media",
            origin_manifest="asr-leaderboard/tedlium.jsonl+gigaspeech.jsonl+voxpopuli.jsonl",
            count=n,
            salt=f"{variant}:noisy_media",
        )
    )
    rows.extend(
        _create_long_rows(
            long_pool,
            source_root=source_root,
            output_dir=output_dir,
            variant=variant,
            count=args.long_samples,
            salt=f"{variant}:long",
        )
    )
    rows.extend(_hallucination_rows(source_root, variant, n))
    rows.extend(_prompt_robustness_rows(rows, variant, max(1, n // len(ASR_PROMPTS)), f"{variant}:prompt"))
    rows.extend(_audio_instruction_rows(source_root, Path(args.preference_asr_dir), variant, n))
    rows.extend(_context_rows(source_root, variant, max(1, n // 3)))
    rows.extend(_text_rows(source_root, variant, args.text_samples))
    return rows


def _multilingual_rows(source_root: Path, output_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = _english_rows(source_root, output_dir, args, variant="multi")

    n = args.audio_samples * args.multi_multiplier
    languages = [lang.strip() for lang in args.languages.split(",") if lang.strip()]
    fleurs = _load_source(source_root, "fleurs", "asr/test.jsonl")
    covost2 = _load_source(source_root, "covost2", "asr/test.jsonl")
    multilingual_pool = _sample_by_language(fleurs + covost2, n, languages, "multi:clean")

    multilingual_rows = []
    for row in multilingual_pool:
        language = (row.get("extra_fields") or {}).get("src_lang") or "unknown"
        multilingual_rows.append(
            _with_metadata(
                row,
                variant="multi",
                subtask="asr.clean_multilingual",
                origin_dataset="fleurs/covost2",
                origin_split="test",
                origin_manifest="fleurs/asr/test.jsonl+covost2/asr/test.jsonl",
                language=language,
            )
        )
    rows.extend(multilingual_rows)
    rows.extend(
        _prompt_robustness_rows(
            multilingual_rows,
            "multi",
            max(1, min(len(multilingual_rows), n // len(ASR_PROMPTS))),
            "multi:prompt",
        )
    )
    return rows


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_entries": len(rows), "subtasks": {}, "origins": {}, "languages": {}}
    for row in rows:
        for section, key in [
            ("subtasks", row.get("ntt_subtask", "unknown")),
            ("origins", row.get("origin_dataset", "unknown")),
            ("languages", row.get("language", "unknown")),
        ]:
            summary[section][key] = summary[section].get(key, 0) + 1
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NTT-SMOKE manifests")
    parser.add_argument(
        "--source-data-dir",
        default=os.getenv("NTT_SMOKE_SOURCE_DATA_DIR") or os.getenv("NEMO_SKILLS_DATA_DIR") or DEFAULT_SOURCE_DATA_DIR,
        help="Prepared source benchmark root containing asr-leaderboard, fleurs, covost2, contextasr-bench, etc.",
    )
    parser.add_argument(
        "--preference-asr-dir",
        default=os.getenv("NTT_SMOKE_PREFERENCE_ASR_DIR") or DEFAULT_PREFERENCE_ASR_DIR,
        help="Optional preference-ASR benchmark root. Currently recorded when present; LibriSpeech-PC is the fallback.",
    )
    parser.add_argument("--audio-samples", type=int, default=200, help="Target sample count per regular audio subtest.")
    parser.add_argument("--text-samples", type=int, default=5000, help="Target sample count for superficial text.")
    parser.add_argument("--long-samples", type=int, default=6, help="Number of stitched long-form English samples.")
    parser.add_argument("--multi-multiplier", type=int, default=2, help="Multilingual size multiplier.")
    parser.add_argument("--short-max-seconds", type=float, default=3.0, help="Maximum duration for very short ASR.")
    parser.add_argument("--languages", default=",".join(DEFAULT_LANGUAGES), help="Comma-separated multilingual locales.")
    parser.add_argument("--output-dir", default=None, help="Override output directory. Defaults to this dataset package.")
    parser.add_argument(
        "--source-read-limit",
        type=int,
        default=0,
        help="Read at most this many lines per source manifest. Intended for local smoke tests; default reads all.",
    )
    parser.add_argument("--skip-multi", action="store_true", help="Only write the English variant.")
    args = parser.parse_args()

    global SOURCE_READ_LIMIT
    SOURCE_READ_LIMIT = args.source_read_limit

    source_root = Path(args.source_data_dir)
    if not source_root.exists():
        raise FileNotFoundError(f"Source data root does not exist: {source_root}")

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    (output_dir / "data").mkdir(parents=True, exist_ok=True)

    en_rows = _english_rows(source_root, output_dir, args)
    en_count = _write_jsonl(output_dir / "en" / "test.jsonl", en_rows)

    summaries = {"en": _summarize(en_rows)}
    print(f"Wrote ntt-smoke.en: {en_count} samples")

    if not args.skip_multi:
        multi_rows = _multilingual_rows(source_root, output_dir, args)
        multi_count = _write_jsonl(output_dir / "multi" / "test.jsonl", multi_rows)
        summaries["multi"] = _summarize(multi_rows)
        print(f"Wrote ntt-smoke.multi: {multi_count} samples")

    pref_dir = Path(args.preference_asr_dir)
    summaries["preference_asr_dir"] = str(pref_dir)
    summaries["preference_asr_available"] = pref_dir.exists()
    summaries["source_data_dir"] = str(source_root)
    _write_jsonl(output_dir / "manifest_summary.jsonl", [{"ntt_smoke_summary": summaries}])

    if not pref_dir.exists():
        print(f"Preference-ASR directory not found; used LibriSpeech-PC fallback: {pref_dir}")


if __name__ == "__main__":
    main()
