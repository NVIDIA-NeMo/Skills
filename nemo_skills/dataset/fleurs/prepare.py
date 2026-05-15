# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
from tqdm import tqdm


def load_fleurs_module():
    """Download and dynamically import google/fleurs/fleurs.py from HuggingFace."""
    import importlib.util

    path = hf_hub_download(repo_id="google/fleurs", filename="fleurs.py", repo_type="dataset")
    spec = importlib.util.spec_from_file_location("fleurs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._FLEURS_LANG, mod._FLEURS_LANG_TO_LONG, mod._FLEURS_LANG_TO_GROUP


FLEURS_LANGS, FLEURS_LANG_TO_LONG, FLEURS_LANG_TO_GROUP = load_fleurs_module()
LOCALES = set(FLEURS_LANGS)

CER_LOCALES = {
    "cmn_hans_cn",  # Mandarin Chinese (Simplified)
    "yue_hant_hk",  # Cantonese Chinese (Traditional)
    "ja_jp",  # Japanese
    "th_th",  # Thai
    "lo_la",  # Lao
    "my_mm",  # Burmese
    "km_kh",  # Khmer
    "ko_kr",  # Korean
    "vi_vn",  # Vietnamese
}


def parse_tsv(tsv_path: str) -> dict[str, dict]:
    """Parse FLEURS TSV metadata file. Returns dict keyed by audio filename."""
    metadata = {}
    with open(tsv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row_id, wav_filename, raw_transcription, transcription, _, _, _ = line.split("\t")
            metadata[wav_filename] = {
                "id": int(row_id),
                "raw_transcription": raw_transcription,
                "transcription": transcription,
                "wav_filename": wav_filename,
            }
    return metadata


def load_fleurs(locale: str, split: str, local_dir: str) -> list[dict]:
    """Download and parse a FLEURS locale/split directly from HuggingFace."""
    tsv_path = hf_hub_download(
        repo_id="google/fleurs",
        filename=f"data/{locale}/{split}.tsv",
        repo_type="dataset",
        local_dir=local_dir,
    )
    tar_path = hf_hub_download(
        repo_id="google/fleurs",
        filename=f"data/{locale}/audio/{split}.tar.gz",
        repo_type="dataset",
        local_dir=local_dir,
    )

    metadata = parse_tsv(tsv_path)
    rows = []
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            wav_filename = Path(member.name).name
            if wav_filename not in metadata:
                continue
            audio_bytes = tar.extractfile(member).read()
            y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            row = dict(metadata[wav_filename])
            row["audio"] = {"array": y, "sampling_rate": sr}
            rows.append(row)
    return rows


def index_by_id(rows: list[dict]) -> dict[int, dict]:
    return {row["id"]: row for row in rows}


def build_translation_pairs(languages: list[str]) -> list[tuple[str, str]]:
    """Build (en_us -> lang) and (lang -> en_us) pairs for each language."""
    pairs = set()
    for lang in languages:
        if lang == "en_us":
            continue
        pairs.add(("en_us", lang))
        pairs.add((lang, "en_us"))
    return sorted(pairs)


def prepare_audio(item: dict) -> tuple[np.ndarray, int, float]:
    audio_dict = item["audio"]
    y, sr = audio_dict["array"], audio_dict["sampling_rate"]
    duration = float(len(y) / sr)
    return y, sr, duration


def get_container_audio_path(locale: str, wav_filename: str) -> str:
    return f"/dataset/fleurs/audio/{locale}/{wav_filename}"


def save_audio(y: np.ndarray, sr: int, wav_path: Path) -> None:
    if not wav_path.exists():
        sf.write(str(wav_path), y, sr)


def get_st_instruction(target_locale: str) -> str:
    tgt_lang_name = FLEURS_LANG_TO_LONG[target_locale]
    return f"Please translate the given speech to {tgt_lang_name}."


def get_asr_instruction() -> str:
    return "Transcribe the following audio."


def _build_record(
    expected_answer: str,
    instruction: str,
    container_audio_path: str,
    duration: float,
    subset_for_metrics: str,
    task_type: str,
    extra_fields: dict,
    source: str | None = None,
    reference: str | None = None,
) -> dict:
    audio_metadata = {"path": container_audio_path, "duration": duration}
    record = {
        "expected_answer": expected_answer,
        "audio_path": container_audio_path,
        "duration": duration,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. /no_think"},
            {"role": "user", "content": instruction, "audio": audio_metadata},
        ],
        "subset_for_metrics": subset_for_metrics,
        "task_type": f"Multilingual-{task_type.upper()}",
        "extra_fields": extra_fields,
    }
    if source is not None:
        record["source"] = source
    if reference is not None:
        record["reference"] = reference
    return record


def _src_extra_fields(source_row: dict, src_locale: str) -> dict:
    return {
        "src_text": source_row["transcription"],
        "src_raw_text": source_row["raw_transcription"],
        "src_lang_name": FLEURS_LANG_TO_LONG[src_locale],
        "src_lang": src_locale,
        "src_lang_group": FLEURS_LANG_TO_GROUP[src_locale],
        "use_cer": src_locale in CER_LOCALES,
    }


def _collect_asr_records(
    languages: list[str],
    audio_dir: Path,
    local_dir: Path,
    split: str,
    no_audio: bool,
) -> list[dict]:
    records: list[dict] = []
    for src_locale in languages:
        locale_audio_dir = audio_dir / src_locale
        if not no_audio:
            locale_audio_dir.mkdir(parents=True, exist_ok=True)
        for source_row in tqdm(load_fleurs(src_locale, split, local_dir=local_dir), desc=src_locale):
            y, sr, duration = prepare_audio(source_row)
            wav_filename = source_row["wav_filename"]
            wav_path = locale_audio_dir / wav_filename
            cpath = get_container_audio_path(src_locale, wav_filename)
            if not no_audio:
                save_audio(y, sr, wav_path)
            records.append(
                _build_record(
                    expected_answer=source_row["transcription"],
                    instruction=get_asr_instruction(),
                    container_audio_path=cpath,
                    duration=duration,
                    subset_for_metrics=src_locale,
                    task_type="ASR",
                    extra_fields=_src_extra_fields(source_row, src_locale),
                )
            )
    return records


def _collect_st_records(
    languages: list[str],
    audio_dir: Path,
    local_dir: Path,
    split: str,
    no_audio: bool,
) -> list[dict]:
    pairs = build_translation_pairs(languages)
    target_cache: dict[str, dict[int, dict]] = {}

    def get_target_index(locale: str) -> dict[int, dict]:
        if locale not in target_cache:
            target_cache[locale] = index_by_id(load_fleurs(locale, split, local_dir=local_dir))
        return target_cache[locale]

    records: list[dict] = []
    for src_locale, tgt_locale in pairs:
        locale_audio_dir = audio_dir / src_locale
        if not no_audio:
            locale_audio_dir.mkdir(parents=True, exist_ok=True)
        target_by_id = get_target_index(tgt_locale)
        tag = f"{src_locale}->{tgt_locale}"
        for source_row in tqdm(load_fleurs(src_locale, split, local_dir=local_dir), desc=tag):
            target_row = target_by_id.get(source_row["id"])
            if target_row is None:
                continue
            y, sr, duration = prepare_audio(source_row)
            wav_filename = source_row["wav_filename"]
            wav_path = locale_audio_dir / wav_filename
            cpath = get_container_audio_path(src_locale, wav_filename)
            if not no_audio:
                save_audio(y, sr, wav_path)
            extra = _src_extra_fields(source_row, src_locale)
            extra.update(
                {
                    "tgt_text": target_row["transcription"],
                    "tgt_raw_text": target_row["raw_transcription"],
                    "tgt_lang_name": FLEURS_LANG_TO_LONG[tgt_locale],
                    "tgt_lang": tgt_locale,
                    "tgt_lang_group": FLEURS_LANG_TO_GROUP[tgt_locale],
                }
            )
            records.append(
                _build_record(
                    expected_answer=target_row["raw_transcription"],
                    instruction=get_st_instruction(tgt_locale),
                    container_audio_path=cpath,
                    duration=duration,
                    subset_for_metrics=tag,
                    task_type="AST",
                    extra_fields=extra,
                    source=source_row["raw_transcription"],
                    reference=target_row["raw_transcription"],
                )
            )
    return records


def _dump_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_fleurs(data_dir: Path, split: str, languages: list[str], no_audio: bool) -> None:
    if not languages:
        raise ValueError("No languages to process")

    audio_dir = data_dir / "audio"
    local_dir = data_dir / "hf-fleurs"
    local_dir.mkdir(parents=True, exist_ok=True)

    asr_dir = data_dir / "asr"
    st_dir = data_dir / "st"
    asr_dir.mkdir(parents=True, exist_ok=True)
    st_dir.mkdir(parents=True, exist_ok=True)
    asr_jsonl = asr_dir / f"{split}.jsonl"
    st_jsonl = st_dir / f"{split}.jsonl"

    asr_records = _collect_asr_records(languages, audio_dir, local_dir, split, no_audio)
    st_records = _collect_st_records(languages, audio_dir, local_dir, split, no_audio)

    _dump_jsonl(asr_jsonl, asr_records)
    _dump_jsonl(st_jsonl, st_records)

    print(f"Fleurs ASR dataset prepared: {asr_jsonl} ({len(asr_records)} records)")
    print(f"Fleurs ST dataset prepared: {st_jsonl} ({len(st_records)} records)")


def main():
    parser = argparse.ArgumentParser(description="Prepare FLEURS Benchmark")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Output directory (defaults to $NEMO_SKILLS_DATA_DIR/fleurs or this package directory)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "dev", "test"],
        help="Dataset split to process",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=LOCALES,
        help="Languages to process",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip saving audio files (only create manifests)",
    )
    args = parser.parse_args()

    unknown = set(args.languages) - LOCALES
    if unknown:
        raise ValueError(f"Unknown language(s): {', '.join(sorted(unknown))}. Available: {', '.join(sorted(LOCALES))}")

    if args.data_dir:
        data_dir = Path(args.data_dir) / "fleurs"
    else:
        data_dir = Path(__file__).parent
    data_dir.mkdir(parents=True, exist_ok=True)

    prepare_fleurs(
        data_dir=data_dir,
        split=args.split,
        languages=args.languages,
        no_audio=args.no_audio,
    )


if __name__ == "__main__":
    main()
