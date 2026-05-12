# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""NeMo-Skills generation wrapper for RTMT-ASR multispeaker SOT inference."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import field
from pathlib import Path
from typing import Any

import hydra

from nemo_skills.utils import get_help_message, nested_dataclass, setup_logging

@nested_dataclass(kw_only=True)
class RTMTASRGenerationConfig:
    """Configuration for RTMT-ASR script-backed generation."""

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
    model_path: str = ""
    nemo_root: str = ""
    script_path: str = ""
    spk_supervision: str = "rttm"
    batch_size: int = 200
    cuda: int = 0
    amp: bool = True
    clean_groundtruth_text: bool = True
    calculate_cpwer: bool = False
    source_lang: str = "en"
    target_lang: str = "en"
    task: str = "asr"
    pnc: str = "no"

    def __post_init__(self):
        pass


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="rtmtasr_generation_config", node=RTMTASRGenerationConfig)


def resolve_script_path(cfg: RTMTASRGenerationConfig) -> Path:
    """Resolve the RTMT-ASR transcribe script path."""
    if cfg.script_path:
        return Path(cfg.script_path)
    root = Path(cfg.nemo_root or os.environ.get("NEMO_ROOT", "."))
    return root / "examples/asr/transcribe_speech_rtmtasr.py"


def build_rtmtasr_command(cfg: RTMTASRGenerationConfig) -> list[str]:
    """Build the RTMT-ASR command from config."""
    if not cfg.model_path:
        raise ValueError("RTMT-ASR generation requires ++model_path=/path/to/model.nemo")
    return [
        sys.executable,
        "-u",
        str(resolve_script_path(cfg)),
        f"model_path={cfg.model_path}",
        "pretrained_name=null",
        "audio_dir=null",
        f"dataset_manifest={cfg.input_file}",
        f"output_filename={cfg.output_file}",
        f"clean_groundtruth_text={str(cfg.clean_groundtruth_text)}",
        "langid=en",
        "gt_lang_attr_name=target_lang",
        f"batch_size={cfg.batch_size}",
        "timestamps=False",
        "compute_langs=False",
        f"cuda={cfg.cuda}",
        f"amp={str(cfg.amp)}",
        "append_pred=False",
        "calculate_wer=False",
        f"calculate_cpwer={str(cfg.calculate_cpwer)}",
        f"spk_supervision={cfg.spk_supervision}",
        f"+prompt.source_lang={cfg.source_lang}",
        f"+prompt.target_lang={cfg.target_lang}",
        f"+prompt.task={cfg.task}",
        f"+prompt.pnc={cfg.pnc}",
    ]


def normalize_prediction_file(path: str | Path) -> None:
    """Ensure RTMT-ASR outputs contain both pred_text and generation."""
    output_path = Path(path)
    rows = []
    with open(output_path, "rt", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            row = json.loads(line)
            pred_text = row.get("pred_text") or row.get("generation") or ""
            row["pred_text"] = pred_text
            row["generation"] = row.get("generation") or pred_text
            rows.append(row)
    with open(output_path, "wt", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row) + "\n")


class RTMTASRGenerationTask:
    """GenerationTask-compatible adapter that runs NeMo's RTMT-ASR script."""

    def __init__(self, cfg: RTMTASRGenerationConfig):
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
        subprocess.run(build_rtmtasr_command(self.cfg), check=True, env=env)
        normalize_prediction_file(self.cfg.output_file)
        if self.cfg.eval_type:
            from nemo_skills.evaluation.evaluator import evaluate

            eval_config = dict(self.cfg.eval_config)
            eval_config["input_file"] = self.cfg.output_file
            evaluate(self.cfg.eval_type, eval_config)


GENERATION_TASK_CLASS = RTMTASRGenerationTask


@hydra.main(version_base=None, config_name="rtmtasr_generation_config")
def generate(cfg: RTMTASRGenerationConfig) -> None:
    cfg = RTMTASRGenerationConfig(_init_nested=True, **cfg)
    task = RTMTASRGenerationTask(cfg)
    task.generate()


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        print(get_help_message(RTMTASRGenerationConfig))
    else:
        setup_logging()
        generate()
