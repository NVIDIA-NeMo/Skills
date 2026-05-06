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
import csv
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

COVOST_URL_TEMPLATE = "https://dl.fbaipublicfiles.com/covost/covost_v2.{src_lang}_{tgt_lang}.tsv.tar.gz"
SPLITS = ["validation", "test"]

LANG_TO_NAME = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ca": "Catalan",
    "it": "Italian",
    "ru": "Russian",
    "zh-CN": "Chinese",
    "pt": "Portuguese",
    "fa": "Persian",
    "et": "Estonian",
    "mn": "Mongolian",
    "nl": "Dutch",
    "tr": "Turkish",
    "ar": "Arabic",
    "sv-SE": "Swedish",
    "lv": "Latvian",
    "sl": "Slovenian",
    "ta": "Tamil",
    "ja": "Japanese",
    "id": "Indonesian",
    "cy": "Welsh",
}

CER_LOCALES = set(["zh-CN", "ja"])

XX_EN_LANGUAGES = [
    "fr",
    "de",
    "es",
    "ca",
    "it",
    "ru",
    "zh-CN",
    "pt",
    "fa",
    "et",
    "mn",
    "nl",
    "tr",
    "ar",
    "sv-SE",
    "lv",
    "sl",
    "ta",
    "ja",
    "id",
    "cy",
]
EN_XX_LANGUAGES = ["de", "tr", "fa", "sv-SE", "mn", "zh-CN", "cy", "ca", "sl", "et", "id", "ar", "ta", "lv", "ja"]
VALID_PAIRS = sorted([(lang, "en") for lang in XX_EN_LANGUAGES] + [("en", lang) for lang in EN_XX_LANGUAGES])
ALL_LANGUAGES = set(["en"]) | set(XX_EN_LANGUAGES) | set(EN_XX_LANGUAGES)


def load_tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\"))


def download_covost_tsv(src_lang: str, tgt_lang: str, local_dir: Path) -> Path:
    tsv_filename = f"covost_v2.{src_lang}_{tgt_lang}.tsv"
    tsv_path = local_dir / tsv_filename
    if tsv_path.exists():
        return tsv_path
    url = COVOST_URL_TEMPLATE.format(src_lang=src_lang, tgt_lang=tgt_lang)
    tar_path = local_dir / f"{tsv_filename}.tar.gz"
    urllib.request.urlretrieve(url, str(tar_path))
    with tarfile.open(str(tar_path), "r:gz") as tar:
        tar.extractall(str(local_dir), filter="data")
    tar_path.unlink()
    return tsv_path


def load_validated_sentences(path: Path) -> dict:
    lookup = {}
    for row in load_tsv(path):
        lookup[(row["path"], row["split"], row["lang"])] = row["sentence"]
    return lookup


def load_covost2(
    src_lang: str,
    tgt_lang: str,
    split: str,
    cv_data_dir: Path,
    local_dir: Path,
    sentences: dict[tuple[str, str, str], str],
) -> list[dict]:
    covost_tsv_path = download_covost_tsv(src_lang, tgt_lang, local_dir)
    covost_split = "dev" if split == "validation" else "test"

    audio_split_dir = cv_data_dir / src_lang / split
    items = []
    for row in load_tsv(covost_tsv_path):
        if row["split"] != covost_split:
            continue
        wav_name = row["path"].replace(".mp3", ".wav")
        wav_file = audio_split_dir / wav_name
        sentence = sentences[(wav_name, split, src_lang)]
        items.append(
            {
                "id": wav_file.stem,
                "sentence": sentence,
                "translation": row["translation"],
                "audio_file": str(wav_file),
            }
        )
    return items


def get_audio_duration(audio_file: str) -> float:
    info = sf.info(audio_file)
    return float(info.frames / info.samplerate)


def get_container_audio_path(src_lang: str, split: str, audio_id: str) -> str:
    return f"/dataset/covost2/audio/{src_lang}/{split}/{audio_id}.wav"


def copy_audio_file(src_wav: Path, audio_dir: Path, src_lang: str, split: str) -> Path:
    dest = audio_dir / src_lang / split / src_wav.name
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_wav, dest)
    return dest


def get_ast_instruction(target_lang: str) -> str:
    tgt_lang_name = LANG_TO_NAME[target_lang]
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
) -> dict:
    audio_metadata = {"path": container_audio_path, "duration": duration}
    return {
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


def prepare_covost2(
    data_dir: Path,
    split: str,
    languages: list[str],
    cv_data_dir: Path,
    validated_tsv: Path,
    task_type: str,
) -> None:
    if not languages:
        raise ValueError("No languages to process")

    if task_type == "ASR":
        pairs = [(lang, lang) for lang in languages]
    elif task_type == "AST":
        lang_set = set(languages)
        pairs = [p for p in VALID_PAIRS if set(p) & lang_set]
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    if not pairs:
        raise ValueError("No (source, target) pairs to process")

    output_jsonl = data_dir / f"{split}-{task_type.lower()}.jsonl"
    sentences = load_validated_sentences(validated_tsv)

    audio_dir = data_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    if task_type == "AST":
        local_dir = data_dir / "fb-covost2"
        local_dir.mkdir(parents=True, exist_ok=True)

    with open(output_jsonl, "w", encoding="utf-8") as out:
        for src_lang, tgt_lang in pairs:
            if task_type == "ASR":
                tag = src_lang
                audio_split_dir = cv_data_dir / src_lang / split
                wav_files = sorted(audio_split_dir.glob("*.wav"))
                for wav_file in tqdm(wav_files, desc=tag):
                    sentence = sentences[(wav_file.name, split, src_lang)]
                    duration = get_audio_duration(str(wav_file))
                    copy_audio_file(wav_file, audio_dir, src_lang, split)
                    cpath = get_container_audio_path(src_lang, split, wav_file.stem)
                    record = _build_record(
                        expected_answer=sentence,
                        instruction=get_asr_instruction(),
                        container_audio_path=cpath,
                        duration=duration,
                        subset_for_metrics=src_lang,
                        task_type=task_type,
                        extra_fields={
                            "src_text": sentence,
                            "src_lang_name": LANG_TO_NAME[src_lang],
                            "src_lang": src_lang,
                            "use_cer": src_lang in CER_LOCALES,
                        },
                    )
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                tag = f"{src_lang}->{tgt_lang}"
                dataset = load_covost2(src_lang, tgt_lang, split, cv_data_dir, local_dir, sentences)
                for item in tqdm(dataset, desc=tag):
                    duration = get_audio_duration(item["audio_file"])
                    copy_audio_file(Path(item["audio_file"]), audio_dir, src_lang, split)
                    cpath = get_container_audio_path(src_lang, split, item["id"])
                    record = _build_record(
                        expected_answer=item["translation"],
                        instruction=get_ast_instruction(tgt_lang),
                        container_audio_path=cpath,
                        duration=duration,
                        subset_for_metrics=tag,
                        task_type=task_type,
                        extra_fields={
                            "src_text": item["sentence"],
                            "tgt_text": item["translation"],
                            "src_lang_name": LANG_TO_NAME[src_lang],
                            "tgt_lang_name": LANG_TO_NAME[tgt_lang],
                            "src_lang": src_lang,
                            "tgt_lang": tgt_lang,
                        },
                    )
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"CoVoST2 {task_type} dataset prepared: {output_jsonl}")


def main():
    parser = argparse.ArgumentParser(description="Prepare CoVoST2 Speech Translation Benchmark")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Output directory (defaults to $NEMO_SKILLS_DATA_DIR/covost2 or this package directory)",
    )
    parser.add_argument(
        "--cv_data_dir",
        type=str,
        required=True,
        help=(
            "Path to audio root directory. Expected layout: <cv_data_dir>/<lang>/<split>/common_voice_<lang>_<id>.wav"
        ),
    )
    parser.add_argument(
        "--validated_tsv",
        type=str,
        required=True,
        help="Path to validated.tsv (columns: path, split, lang, sentence)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=SPLITS,
        help="Dataset split to process",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=ALL_LANGUAGES,
        default=ALL_LANGUAGES,
        help="Languages to process (all valid pairs with English are included)",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["ASR", "AST", "asr", "ast"],
        help="Task to prepare. ASR: speech recognition, AST: speech translation.",
    )
    args = parser.parse_args()

    if args.data_dir:
        data_dir = Path(args.data_dir) / "covost2"
    else:
        data_dir = Path(__file__).parent
    data_dir.mkdir(parents=True, exist_ok=True)

    prepare_covost2(
        data_dir=data_dir,
        split=args.split,
        languages=args.languages,
        cv_data_dir=Path(args.cv_data_dir),
        validated_tsv=Path(args.validated_tsv),
        task_type=args.task.upper(),
    )


if __name__ == "__main__":
    main()
