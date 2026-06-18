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

"""Prepare NTT-COMPLEX manifests from already-prepared NeMo-Skills data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

SYSTEM_MESSAGE = "You are a helpful assistant. /no_think"
DEFAULT_SOURCE_DATA_DIR = (
    "/home/vmendelev/.cache/saferun/cluster-sshfs/oci_iad/lustre/fs12/portfolios/llmservice/users/"
    "pzelasko/results/speechlm-2026h1/skills_data"
)

FORMAT_SPECS = {
    "json_object": {
        "name": "JSON object",
        "prompt": (
            "Translate the speech to {target_language}. Respond with only a valid JSON object "
            'with exactly these keys: "source_language", "target_language", "translation". '
            'Set "translation" to the translated speech.'
        ),
    },
    "srt_single_cue": {
        "name": "single-cue SRT",
        "prompt": (
            "Translate the speech to {target_language}. Respond with only one SRT subtitle cue: "
            "line 1 must be the cue number 1, line 2 must be "
            "00:00:00,000 --> 00:00:05,000, and the remaining line must be the translation."
        ),
    },
    "markdown_table": {
        "name": "Markdown table",
        "prompt": (
            "Translate the speech to {target_language}. Respond with only a Markdown table "
            "with columns source_language, target_language, translation, and exactly one data row."
        ),
    },
}

SOURCE_MANIFEST_CANDIDATES = {
    "fleurs": ("st/test.jsonl", "ast/test.jsonl"),
    "covost2": ("st/test.jsonl", "ast/test.jsonl"),
}


def _stable_key(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as fin:
        for line in fin:
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
    if count <= 0:
        return []
    keyed = sorted(rows, key=lambda row: _stable_key([salt, row.get("id"), row.get("audio_path"), row]))
    return keyed[: min(count, len(keyed))]


def _origin_id(row: dict[str, Any]) -> str:
    for key in ("id", "sample_id", "uniq_id", "key"):
        if row.get(key) is not None:
            return str(row[key])
    audio_path = row.get("audio_path") or row.get("audio_filepath")
    if isinstance(audio_path, str):
        return Path(audio_path).stem
    return _stable_key(row)[:16]


def _source_language(row: dict[str, Any]) -> str:
    extra_fields = row.get("extra_fields") or {}
    return str(extra_fields.get("src_lang_name") or extra_fields.get("src_lang") or row.get("language") or "unknown")


def _target_language(row: dict[str, Any]) -> str:
    extra_fields = row.get("extra_fields") or {}
    return str(extra_fields.get("tgt_lang_name") or extra_fields.get("tgt_lang") or "target language")


def _target_lang_code(row: dict[str, Any]) -> str | None:
    extra_fields = row.get("extra_fields") or {}
    value = extra_fields.get("tgt_lang")
    return str(value) if value is not None else None


def _reference(row: dict[str, Any]) -> str:
    return str(row.get("reference") or row.get("expected_answer") or "")


def _with_audio_prompt(row: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
    messages = copy.deepcopy(row.get("messages") or [])
    if not any(message.get("role") == "system" for message in messages):
        messages.insert(0, {"role": "system", "content": SYSTEM_MESSAGE})

    audio_meta = None
    for message in messages:
        if message.get("role") == "user" and isinstance(message.get("audio"), dict):
            audio_meta = copy.deepcopy(message["audio"])
            break
    if audio_meta is None:
        audio_path = row.get("audio_path") or row.get("audio_filepath")
        audio_meta = {"path": audio_path}
        if row.get("duration") is not None:
            audio_meta["duration"] = row["duration"]

    for idx, message in enumerate(messages):
        if message.get("role") == "user":
            messages[idx] = {"role": "user", "content": prompt, "audio": audio_meta}
            break
    else:
        messages.append({"role": "user", "content": prompt, "audio": audio_meta})
    return messages


def _render_row(row: dict[str, Any], source_dataset: str, origin_manifest: str, format_id: str) -> dict[str, Any]:
    format_spec = FORMAT_SPECS[format_id]
    source_language = _source_language(row)
    target_language = _target_language(row)
    prompt = format_spec["prompt"].format(target_language=target_language)
    extra_fields = copy.deepcopy(row.get("extra_fields") or {})
    extra_fields.setdefault("src_lang_name", source_language)
    extra_fields.setdefault("tgt_lang_name", target_language)
    if _target_lang_code(row) is not None:
        extra_fields.setdefault("tgt_lang", _target_lang_code(row))

    out = copy.deepcopy(row)
    out.update(
        {
            "task_type": "Format-AST",
            "expected_answer": _reference(row),
            "reference": _reference(row),
            "source": row.get("source"),
            "messages": _with_audio_prompt(row, prompt),
            "subset_for_metrics": f"format_ast.{format_id}",
            "ntt_complex_subtest": "format_ast",
            "format_id": format_id,
            "format_name": format_spec["name"],
            "origin_dataset": source_dataset,
            "origin_manifest": f"{source_dataset}/{origin_manifest}",
            "origin_id": _origin_id(row),
            "extra_fields": extra_fields,
        }
    )
    return out


def _load_source_rows(source_root: Path, source_dataset: str) -> tuple[str, list[dict[str, Any]]]:
    for rel_path in SOURCE_MANIFEST_CANDIDATES[source_dataset]:
        rows = _read_jsonl(source_root / source_dataset / rel_path)
        if rows:
            return rel_path, rows
    return SOURCE_MANIFEST_CANDIDATES[source_dataset][0], []


def build_format_ast_rows(source_root: Path, samples_per_source: int, sources: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source_dataset in sources:
        origin_manifest, source_rows = _load_source_rows(source_root, source_dataset)
        selected = _sample(source_rows, samples_per_source, f"format_ast:{source_dataset}")
        for source_row in selected:
            for format_id in FORMAT_SPECS:
                rows.append(_render_row(source_row, source_dataset, origin_manifest, format_id))
    return sorted(rows, key=lambda row: _stable_key([row["origin_dataset"], row["origin_id"], row["format_id"]]))


def prepare_ntt_complex(
    data_dir: Path,
    source_data_dir: Path,
    samples_per_source: int,
    sources: list[str],
) -> None:
    rows = build_format_ast_rows(source_data_dir, samples_per_source, sources)
    count = _write_jsonl(data_dir / "format_ast" / "test.jsonl", rows)
    _write_jsonl(
        data_dir / "manifest_summary.jsonl",
        [
            {
                "ntt_complex_summary": {
                    "format_ast": {
                        "num_entries": count,
                        "source_data_dir": str(source_data_dir),
                        "sources": sources,
                        "formats": list(FORMAT_SPECS),
                    }
                }
            }
        ],
    )
    print(f"Wrote ntt-complex.format_ast: {count} samples")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NTT-COMPLEX manifests")
    parser.add_argument(
        "--source-data-dir",
        default=os.getenv("NTT_COMPLEX_SOURCE_DATA_DIR") or os.getenv("NEMO_SKILLS_DATA_DIR") or DEFAULT_SOURCE_DATA_DIR,
        help="Prepared source benchmark root containing fleurs and covost2.",
    )
    parser.add_argument("--samples-per-source", type=int, default=200, help="Samples to draw from each source dataset.")
    parser.add_argument(
        "--sources",
        default="fleurs,covost2",
        help="Comma-separated prepared source datasets to use. Supported: fleurs,covost2.",
    )
    parser.add_argument("--output-dir", default=None, help="Override output directory. Defaults to this package directory.")
    args = parser.parse_args()

    sources = [source.strip() for source in args.sources.split(",") if source.strip()]
    unknown_sources = sorted(set(sources) - set(SOURCE_MANIFEST_CANDIDATES))
    if unknown_sources:
        raise ValueError(f"Unsupported source dataset(s): {', '.join(unknown_sources)}")

    source_data_dir = Path(args.source_data_dir)
    if not source_data_dir.exists():
        raise FileNotFoundError(f"Source data root does not exist: {source_data_dir}")

    data_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent
    prepare_ntt_complex(
        data_dir=data_dir,
        source_data_dir=source_data_dir,
        samples_per_source=args.samples_per_source,
        sources=sources,
    )


if __name__ == "__main__":
    main()
