# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Evaluator for LibriSpeechMix SOT cpWER."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import tqdm

from nemo_skills.evaluation.evaluator.base import BaseEvaluator, BaseEvaluatorConfig
from nemo_skills.evaluation.metrics.librispeechmix_sot_utils import cpwer, write_meeteval_artifacts
from nemo_skills.utils import nested_dataclass


@nested_dataclass(kw_only=True)
class LibriSpeechMixSOTEvaluatorConfig(BaseEvaluatorConfig):
    """Configuration for LibriSpeechMix SOT evaluation."""

    prediction_keys: list[str] | None = None
    write_meeteval: bool = True


class LibriSpeechMixSOTEvaluator(BaseEvaluator):
    """Compute per-record cpWER and optional MeetEval artifacts."""

    def __init__(self, config: Dict[str, Any], num_parallel_requests=10):
        super().__init__(LibriSpeechMixSOTEvaluatorConfig(_init_nested=True, **config), num_parallel_requests)

    def _get_hypothesis(self, data_point: dict[str, Any]) -> str:
        keys = self.config.prediction_keys or ["pred_text", "generation"]
        for key in keys:
            value = data_point.get(key)
            if value is not None:
                return str(value)
        return ""

    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        hypothesis = self._get_hypothesis(data_point)
        result = cpwer(data_point["text"], hypothesis)
        return {
            "pred_text": hypothesis,
            "cpwer": result["cpwer"],
            "cpwer_errors": result["errors"],
            "cpwer_substitutions": result["substitutions"],
            "cpwer_insertions": result["insertions"],
            "cpwer_deletions": result["deletions"],
            "cpwer_ref_words": result["ref_words"],
            "speaker_assignment": result["assignment"],
            "is_correct": result["errors"] == 0,
        }

    async def eval_full(self) -> None:
        records = []
        with open(self.config.input_file, "rt", encoding="utf-8") as fin:
            for line in tqdm.tqdm(fin, desc=f"Evaluating {os.path.basename(self.config.input_file)}"):
                record = json.loads(line)
                record.update(await self.eval_single(record))
                if "generation" not in record:
                    record["generation"] = record["pred_text"]
                records.append(record)

        temp_file = self.config.input_file + "-tmp"
        with open(temp_file, "wt", encoding="utf-8") as fout:
            for record in records:
                fout.write(json.dumps(record) + "\n")
        os.replace(temp_file, self.config.input_file)

        if self.config.write_meeteval:
            artifacts = write_meeteval_artifacts(records, self.config.input_file)
            sidecar = self.config.input_file + ".meeteval_artifacts.json"
            with open(sidecar, "wt", encoding="utf-8") as fout:
                json.dump(artifacts, fout, indent=2)
