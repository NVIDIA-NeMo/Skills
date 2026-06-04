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

"""Validate Parakeet v3 on AppTek Call-Center Dialogues.

By default this script transcribes each full-channel WAV directly with
``nvidia/parakeet-tdt-0.6b-v3`` and enables the model-card local-attention
setting for long audio. Use ``--mode silero`` to reproduce the AppTek paper's
Silero-segmented condition.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import jiwer
from tqdm import tqdm

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v3"
SAMPLING_RATE = 16_000


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _default_work_dir() -> Path:
    repo_root = _repo_root()
    return repo_root.parent / f"{repo_root.name}-data" / "apptek-parakeet-v3-repro"


def _default_manifest() -> Path:
    return _repo_root() / "nemo_skills" / "dataset" / "apptek-callcenter-dialogues" / "test.jsonl"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fin:
        return [json.loads(line) for line in fin if line.strip()]


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def _row_extra(row: dict[str, Any], key: str) -> Any:
    """Read a per-row metadata value from extra_fields, falling back to top level."""
    extras = row.get("extra_fields") or {}
    if key in extras:
        return extras[key]
    return row.get(key)


def _row_file_name(row: dict[str, Any]) -> str:
    return _row_extra(row, "file_name")


def _row_accent_code(row: dict[str, Any]) -> str:
    return _row_extra(row, "accent_code") or row.get("subset_for_metrics") or "unknown"


def _manifest_audio_path(row: dict[str, Any]) -> Path:
    path = row.get("audio_filepath")
    if not path:
        for message in row.get("messages", []):
            audio = message.get("audio")
            if audio:
                path = audio["path"]
                break
    if not path:
        raise ValueError(f"No audio path found for row with file_name={_row_file_name(row)}")
    return Path(path).expanduser().resolve()


def _segment_path(segment_root: Path, file_name: str, segment_index: int) -> Path:
    rel = Path(file_name)
    return segment_root / rel.parent / f"{rel.stem}__seg{segment_index:04d}.wav"


def _load_existing_segments(path: Path) -> dict[str, list[dict[str, Any]]]:
    if not path.exists():
        return {}
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(path):
        by_file[row["source_file_name"]].append(row)
    for rows in by_file.values():
        rows.sort(key=lambda item: item["segment_index"])
    return dict(by_file)


def segment_manifest(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create or reuse Silero VAD segments for all manifest rows."""
    import silero_vad
    import soundfile as sf

    segments_path = args.work_dir / "segments.jsonl"
    existing = _load_existing_segments(segments_path)
    complete = {
        file_name
        for file_name, file_segments in existing.items()
        if all(Path(segment["segment_audio_filepath"]).exists() for segment in file_segments)
    }

    missing_rows = [row for row in rows if _row_file_name(row) not in complete]
    if not args.force_segment and not missing_rows:
        return [segment for file_segments in existing.values() for segment in file_segments]

    if args.force_segment and segments_path.exists():
        segments_path.unlink()
        existing = {}
        missing_rows = rows

    model = silero_vad.load_silero_vad(onnx=False)
    new_segments: list[dict[str, Any]] = []
    segment_root = args.work_dir / "segments"

    for row in tqdm(missing_rows, desc="Silero segmentation"):
        file_name = _row_file_name(row)
        audio_path = _manifest_audio_path(row)
        if not audio_path.exists():
            raise FileNotFoundError(f"Missing audio for {file_name}: {audio_path}")

        audio = silero_vad.read_audio(str(audio_path), sampling_rate=SAMPLING_RATE)
        timestamps = silero_vad.get_speech_timestamps(
            audio,
            model,
            sampling_rate=SAMPLING_RATE,
            min_silence_duration_ms=args.min_silence_duration_ms,
            min_speech_duration_ms=args.min_speech_duration_ms,
            max_speech_duration_s=args.max_speech_duration_s,
            speech_pad_ms=args.speech_pad_ms,
            return_seconds=False,
        )

        file_segments = []
        for segment_index, timestamp in enumerate(timestamps):
            start_sample = int(timestamp["start"])
            end_sample = int(timestamp["end"])
            segment_audio = audio[start_sample:end_sample].detach().cpu().numpy()
            path = _segment_path(segment_root, file_name, segment_index)
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(path), segment_audio, SAMPLING_RATE)
            file_segments.append(
                {
                    "source_file_name": file_name,
                    "source_audio_filepath": str(audio_path),
                    "segment_audio_filepath": str(path),
                    "segment_index": segment_index,
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "start_sec": start_sample / SAMPLING_RATE,
                    "end_sec": end_sample / SAMPLING_RATE,
                    "accent_code": _row_accent_code(row),
                }
            )

        _append_jsonl(segments_path, file_segments)
        existing[file_name] = file_segments
        new_segments.extend(file_segments)

    print(f"Wrote {len(new_segments)} new segments to {segments_path}")
    return [segment for file_segments in existing.values() for segment in file_segments]


def _load_existing_predictions(path: Path, key: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {row[key]: row for row in _read_jsonl(path)}


def _hypothesis_text(hypothesis: Any) -> str:
    if isinstance(hypothesis, str):
        return hypothesis
    if hasattr(hypothesis, "text"):
        return str(hypothesis.text)
    if isinstance(hypothesis, dict) and "text" in hypothesis:
        return str(hypothesis["text"])
    return str(hypothesis)


def _load_asr_model(args: argparse.Namespace):
    import nemo.collections.asr as nemo_asr
    import torch

    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name=args.model_name)

    if args.long_form_attention == "local":
        asr_model.change_attention_model(
            self_attention_model="rel_pos_local_attn",
            att_context_size=list(args.att_context_size),
        )
    if args.subsampling_conv_chunking_factor is not None:
        asr_model.change_subsampling_conv_chunking_factor(args.subsampling_conv_chunking_factor)

    asr_model.eval()
    if args.device:
        asr_model = asr_model.to(args.device)
    elif torch.cuda.is_available():
        asr_model = asr_model.to("cuda")
    return asr_model


def _transcribe_batch(asr_model, audio_paths: list[str], args: argparse.Namespace) -> list[str]:
    hypotheses = asr_model.transcribe(audio_paths, batch_size=args.batch_size, verbose=not args.quiet)
    if isinstance(hypotheses, tuple):
        hypotheses = hypotheses[0]
    return [_hypothesis_text(hypothesis).strip() for hypothesis in hypotheses]


def transcribe_direct(args: argparse.Namespace, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Transcribe full-channel AppTek WAVs directly with Parakeet v3."""
    predictions_path = args.work_dir / "direct_predictions.jsonl"
    existing = _load_existing_predictions(predictions_path, key="file_name")
    missing = [row for row in rows if _row_file_name(row) not in existing]
    if not missing:
        return list(existing.values())

    import torch

    asr_model = _load_asr_model(args)

    with torch.inference_mode():
        for offset in tqdm(range(0, len(missing), args.batch_size), desc="Parakeet direct transcription"):
            batch = missing[offset : offset + args.batch_size]
            audio_paths = [str(_manifest_audio_path(row)) for row in batch]
            texts = _transcribe_batch(asr_model, audio_paths, args)
            prediction_rows = []
            for source_row, text in zip(batch, texts, strict=True):
                row = {
                    "file_name": _row_file_name(source_row),
                    "audio_filepath": str(_manifest_audio_path(source_row)),
                    "text": text,
                    "accent_code": _row_accent_code(source_row),
                }
                prediction_rows.append(row)
                existing[row["file_name"]] = row
            _append_jsonl(predictions_path, prediction_rows)

    print(f"Wrote direct predictions to {predictions_path}")
    predictions = [existing[_row_file_name(row)] for row in rows if _row_file_name(row) in existing]
    output_path = args.work_dir / "predictions.jsonl"
    _write_jsonl(output_path, predictions)
    print(f"Wrote file-level predictions to {output_path}")
    return predictions


def transcribe_segments(args: argparse.Namespace, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Transcribe missing Silero segments with Parakeet v3."""
    predictions_path = args.work_dir / "segment_predictions.jsonl"
    existing = _load_existing_predictions(predictions_path, key="segment_audio_filepath")
    missing = [segment for segment in segments if segment["segment_audio_filepath"] not in existing]
    if not missing:
        return list(existing.values())

    import torch

    asr_model = _load_asr_model(args)

    with torch.inference_mode():
        for offset in tqdm(range(0, len(missing), args.batch_size), desc="Parakeet transcription"):
            batch = missing[offset : offset + args.batch_size]
            audio_paths = [row["segment_audio_filepath"] for row in batch]
            texts = _transcribe_batch(asr_model, audio_paths, args)
            rows = []
            for segment, text in zip(batch, texts, strict=True):
                row = dict(segment)
                row["text"] = text
                rows.append(row)
                existing[row["segment_audio_filepath"]] = row
            _append_jsonl(predictions_path, rows)

    print(f"Wrote segment predictions to {predictions_path}")
    return list(existing.values())


def aggregate_predictions(
    args: argparse.Namespace, rows: list[dict[str, Any]], segment_predictions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Concatenate segment hypotheses into one prediction per source file."""
    by_file: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in segment_predictions:
        by_file[prediction["source_file_name"]].append(prediction)

    output_rows = []
    for row in rows:
        file_name = _row_file_name(row)
        parts = sorted(by_file.get(file_name, []), key=lambda item: item["segment_index"])
        output_rows.append(
            {
                "file_name": file_name,
                "text": " ".join(part["text"] for part in parts if part.get("text")).strip(),
                "accent_code": _row_accent_code(row),
            }
        )

    predictions_path = args.work_dir / "predictions.jsonl"
    _write_jsonl(predictions_path, output_rows)
    print(f"Wrote file-level predictions to {predictions_path}")
    return output_rows


def _score_group(refs: list[str], hyps: list[str]) -> dict[str, Any]:
    measures = jiwer.process_words(refs, hyps)
    errors = measures.substitutions + measures.deletions + measures.insertions
    ref_words = measures.substitutions + measures.deletions + measures.hits
    return {
        "wer": errors / ref_words if ref_words else 0.0,
        "wer_percent": 100 * errors / ref_words if ref_words else 0.0,
        "wer_errors": errors,
        "wer_ref_words": ref_words,
        "wer_substitutions": measures.substitutions,
        "wer_deletions": measures.deletions,
        "wer_insertions": measures.insertions,
    }


def score_predictions(args: argparse.Namespace, rows: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict:
    """Score file-level predictions with AppTek reference/prediction normalization."""
    from nemo_skills.evaluation.evaluator.apptek_callcenter import normalize_apptek_callcenter_text

    prediction_by_file = {row["file_name"]: row["text"] for row in predictions}
    refs = []
    hyps = []
    by_accent: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"refs": [], "hyps": []})

    for row in sorted(rows, key=_row_file_name):
        file_name = _row_file_name(row)
        accent_code = _row_accent_code(row)
        ref = normalize_apptek_callcenter_text(row["expected_answer"], is_prediction=False)
        hyp = normalize_apptek_callcenter_text(prediction_by_file.get(file_name, ""), is_prediction=True)
        refs.append(ref)
        hyps.append(hyp)
        by_accent[accent_code]["refs"].append(ref)
        by_accent[accent_code]["hyps"].append(hyp)

    segments_path = args.work_dir / "segments.jsonl"
    metrics = {
        "mode": args.mode,
        "model_name": args.model_name,
        "num_files": len(rows),
        "overall": _score_group(refs, hyps),
        "by_accent": {
            accent: _score_group(items["refs"], items["hyps"]) for accent, items in sorted(by_accent.items())
        },
    }
    if args.mode == "direct":
        metrics["direct"] = {
            "long_form_attention": args.long_form_attention,
            "att_context_size": list(args.att_context_size),
            "subsampling_conv_chunking_factor": args.subsampling_conv_chunking_factor,
        }
    else:
        metrics["num_segments"] = len(_read_jsonl(segments_path)) if segments_path.exists() else 0
        metrics["silero"] = {
            "min_silence_duration_ms": args.min_silence_duration_ms,
            "min_speech_duration_ms": args.min_speech_duration_ms,
            "max_speech_duration_s": args.max_speech_duration_s,
            "speech_pad_ms": args.speech_pad_ms,
        }

    metrics_path = args.work_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote metrics to {metrics_path}")
    print(f"Overall WER: {metrics['overall']['wer_percent']:.2f}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=_default_manifest(), help="Prepared AppTek test.jsonl.")
    parser.add_argument(
        "--work-dir", type=Path, default=_default_work_dir(), help="Output directory outside the repo."
    )
    parser.add_argument("--model-name", default=MODEL_NAME, help="NeMo/HuggingFace ASR model name.")
    parser.add_argument(
        "--device", default=None, help="Torch device, e.g. cuda, cuda:0, or cpu. Defaults to cuda if present."
    )
    parser.add_argument(
        "--mode",
        choices=("direct", "silero"),
        default="direct",
        help="direct transcribes full-channel WAVs; silero reproduces the paper's Silero-segmented condition.",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Parakeet transcription batch size.")
    parser.add_argument(
        "--long-form-attention",
        choices=("local", "unchanged"),
        default="local",
        help="Use Parakeet model-card local attention for long audio, or leave attention unchanged.",
    )
    parser.add_argument(
        "--att-context-size",
        type=int,
        nargs=2,
        default=(256, 256),
        metavar=("LEFT", "RIGHT"),
        help="Attention context used when --long-form-attention=local.",
    )
    parser.add_argument(
        "--subsampling-conv-chunking-factor",
        type=int,
        default=None,
        help="Optional NeMo conv subsampling chunking factor to reduce memory if long-form transcription OOMs.",
    )
    parser.add_argument("--quiet", action="store_true", help="Disable NeMo transcribe progress bars.")
    parser.add_argument("--min-silence-duration-ms", type=int, default=10_000)
    parser.add_argument("--min-speech-duration-ms", type=int, default=250)
    parser.add_argument("--max-speech-duration-s", type=float, default=30.0)
    parser.add_argument("--speech-pad-ms", type=int, default=30)
    parser.add_argument(
        "--max-files", type=int, default=None, help="Debug limit on manifest rows before running stages."
    )
    parser.add_argument("--force-segment", action="store_true", help="Regenerate Silero segments from scratch.")
    parser.add_argument("--skip-transcribe", action="store_true", help="Only segment and score existing predictions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.manifest = args.manifest.expanduser().resolve()
    args.work_dir = args.work_dir.expanduser().resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(args.manifest)
    if args.max_files is not None:
        rows = rows[: args.max_files]

    if args.mode == "direct":
        predictions = transcribe_direct(args, rows)
    else:
        segments = segment_manifest(args, rows)
        if args.skip_transcribe:
            segment_predictions = list(
                _load_existing_predictions(
                    args.work_dir / "segment_predictions.jsonl", key="segment_audio_filepath"
                ).values()
            )
        else:
            segment_predictions = transcribe_segments(args, segments)
        predictions = aggregate_predictions(args, rows, segment_predictions)

    score_predictions(args, rows, predictions)


if __name__ == "__main__":
    main()
