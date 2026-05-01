# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""NeMo-Skills generation wrapper for Multitalker Parakeet streaming inference."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import field
from pathlib import Path
from typing import Any

import hydra

from nemo_skills.utils import get_help_message, nested_dataclass, setup_logging

DEFAULT_ASR_MODEL = "nvidia/multitalker-parakeet-streaming-0.6b-v1"
DEFAULT_DIAR_MODEL = "nvidia/diar_streaming_sortformer_4spk-v2.1"


@nested_dataclass(kw_only=True)
class MultitalkerParakeetGenerationConfig:
    """Configuration for Multitalker Parakeet script-backed generation."""

    input_file: str
    output_file: str
    prompt_config: Any = None
    prompt_format: str = "openai"
    server: dict = field(default_factory=dict)
    eval_type: str | None = None
    eval_config: dict = field(default_factory=dict)
    skip_filled: bool = False
    num_chunks: int | None = None
    chunk_id: int | None = None
    nemo_root: str = ""
    script_path: str = ""
    asr_model: str = DEFAULT_ASR_MODEL
    diar_model: str = DEFAULT_DIAR_MODEL
    spk_supervision: str = "rttm"
    max_num_of_spks: int = 4
    masked_asr: bool = False
    parallel_speaker_strategy: bool = True
    binary_diar_preds: bool = True
    batch_size: int = 125
    att_context_size: str = "[70,13]"
    spkcache_len: int = 188
    spkcache_refresh_rate: int = 144
    fifo_len: int = 188
    chunk_len: int = 13
    chunk_left_context: int = 1
    chunk_right_context: int = 0
    cache_gating: bool = False
    calculate_cpwer: bool = False
    remove_pnc_for_cpwer: bool = True

    def __post_init__(self):
        pass


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="multitalker_parakeet_generation_config", node=MultitalkerParakeetGenerationConfig)


def resolve_script_path(cfg: MultitalkerParakeetGenerationConfig) -> Path:
    """Resolve the streaming multitalker inference script."""
    if cfg.script_path:
        return Path(cfg.script_path)
    root = Path(cfg.nemo_root or os.environ.get("NEMO_ROOT", "."))
    return root / "examples/asr/asr_cache_aware_streaming/speech_to_text_multitalker_streaming_infer.py"


def build_multitalker_parakeet_command(cfg: MultitalkerParakeetGenerationConfig) -> list[str]:
    """Build the Multitalker Parakeet command from config."""
    return [
        sys.executable,
        str(resolve_script_path(cfg)),
        "log=False",
        f"binary_diar_preds={str(cfg.binary_diar_preds)}",
        f"spk_supervision={cfg.spk_supervision}",
        f"max_num_of_spks={cfg.max_num_of_spks}",
        f"masked_asr={str(cfg.masked_asr)}",
        f"asr_model={cfg.asr_model}",
        f"diar_model={cfg.diar_model}",
        f"parallel_speaker_strategy={str(cfg.parallel_speaker_strategy)}",
        f"att_context_size={cfg.att_context_size}",
        "generate_realtime_scripts=False",
        f"batch_size={cfg.batch_size}",
        f"manifest_file={cfg.input_file}",
        f"output_path={cfg.output_file}",
        f"calculate_cpwer={str(cfg.calculate_cpwer)}",
        f"remove_pnc_for_cpwer={str(cfg.remove_pnc_for_cpwer)}",
        f"cache_gating={str(cfg.cache_gating)}",
        f"spkcache_len={cfg.spkcache_len}",
        f"spkcache_refresh_rate={cfg.spkcache_refresh_rate}",
        f"fifo_len={cfg.fifo_len}",
        f"chunk_len={cfg.chunk_len}",
        f"chunk_left_context={cfg.chunk_left_context}",
        f"chunk_right_context={cfg.chunk_right_context}",
    ]


def _load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "rt", encoding="utf-8") as fin:
        for line in fin:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _session_id(record: dict) -> str:
    return str(record.get("sample_id") or record.get("id") or Path(record.get("audio_filepath", "sample")).stem)


def _seglst_to_sot(entries: list[dict]) -> str:
    tag_by_speaker = {}
    chunks = []
    for entry in sorted(
        entries,
        key=lambda item: (
            float(item.get("start_time", 0.0) or 0.0),
            float(item.get("end_time", 0.0) or 0.0),
            str(item.get("speaker", "")),
        ),
    ):
        words = str(entry.get("words") or entry.get("text") or "").strip()
        if not words:
            continue
        speaker = str(entry.get("speaker") or "speaker")
        if speaker not in tag_by_speaker:
            tag_by_speaker[speaker] = f"s{len(tag_by_speaker)}"
        chunks.append(f"[{tag_by_speaker[speaker]}] {words}")
    return " ".join(chunks)


def _normalize_seglst_output(output_path: Path, input_file: str | Path | None, entries: list[dict]) -> None:
    by_session: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        session_id = str(entry.get("session_id") or entry.get("recording_id") or entry.get("audio_id") or "sample")
        by_session[session_id].append(entry)

    if input_file and Path(input_file).exists():
        rows = _load_jsonl(input_file)
        for row in rows:
            session_id = _session_id(row)
            session_entries = by_session.get(session_id, [])
            pred_text = _seglst_to_sot(session_entries)
            row["pred_text"] = pred_text
            row["generation"] = row.get("generation") or pred_text
            row["hypothesis_seglst"] = session_entries
    else:
        rows = []
        for session_id, session_entries in sorted(by_session.items()):
            pred_text = _seglst_to_sot(session_entries)
            rows.append(
                {
                    "sample_id": session_id,
                    "pred_text": pred_text,
                    "generation": pred_text,
                    "hypothesis_seglst": session_entries,
                }
            )

    with open(output_path, "wt", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row) + "\n")


def normalize_prediction_file(path: str | Path, input_file: str | Path | None = None) -> None:
    """Ensure Multitalker outputs contain both pred_text and generation."""
    output_path = Path(path)
    raw = output_path.read_text(encoding="utf-8").strip()
    if not raw:
        output_path.write_text("", encoding="utf-8")
        return

    parsed = json.loads(raw.splitlines()[0] if raw[0] != "[" else raw)
    if isinstance(parsed, list):
        _normalize_seglst_output(output_path, input_file, parsed)
        return

    rows = []
    with open(output_path, "rt", encoding="utf-8") as fin:
        for line_idx, line in enumerate(fin):
            if not line.strip():
                continue
            row = parsed if line_idx == 0 else json.loads(line)
            pred_text = row.get("pred_text") or row.get("generation") or row.get("text_pred") or ""
            row["pred_text"] = pred_text
            row["generation"] = row.get("generation") or pred_text
            rows.append(row)
    with open(output_path, "wt", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row) + "\n")


class MultitalkerParakeetGenerationTask:
    """GenerationTask-compatible adapter for the streaming multitalker script."""

    def __init__(self, cfg: MultitalkerParakeetGenerationConfig):
        self.cfg = cfg

    @classmethod
    def get_generation_default_args(cls) -> str:
        return "++prompt_format=openai"

    @classmethod
    def get_generation_requirements(cls) -> list[str] | None:
        return None

    def generate(self) -> None:
        Path(self.cfg.output_file).parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        if self.cfg.nemo_root:
            env["PYTHONPATH"] = f"{self.cfg.nemo_root}:{env.get('PYTHONPATH', '')}"
        subprocess.run(build_multitalker_parakeet_command(self.cfg), check=True, env=env)
        normalize_prediction_file(self.cfg.output_file, self.cfg.input_file)
        if self.cfg.eval_type:
            from nemo_skills.evaluation.evaluator import evaluate

            eval_config = dict(self.cfg.eval_config)
            eval_config["input_file"] = self.cfg.output_file
            evaluate(self.cfg.eval_type, eval_config)


GENERATION_TASK_CLASS = MultitalkerParakeetGenerationTask


@hydra.main(version_base=None, config_name="multitalker_parakeet_generation_config")
def generate(cfg: MultitalkerParakeetGenerationConfig) -> None:
    cfg = MultitalkerParakeetGenerationConfig(_init_nested=True, **cfg)
    task = MultitalkerParakeetGenerationTask(cfg)
    task.generate()


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(get_help_message(MultitalkerParakeetGenerationConfig))
    else:
        setup_logging()
        generate()
