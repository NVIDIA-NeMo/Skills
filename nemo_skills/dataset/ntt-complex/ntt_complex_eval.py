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

"""Evaluator for NTT-COMPLEX subtests."""

from __future__ import annotations

import json
import re
from dataclasses import fields
from typing import Any

from nemo_skills.evaluation.evaluator.audio import AudioEvaluatorConfig, evaluate_translation
from nemo_skills.evaluation.evaluator.base import BaseEvaluator


def _audio_config(config: dict[str, Any]) -> AudioEvaluatorConfig:
    field_names = {field.name for field in fields(AudioEvaluatorConfig)}
    return AudioEvaluatorConfig(**{key: value for key, value in config.items() if key in field_names})


def _clean_generation(generation: str, config: AudioEvaluatorConfig) -> str:
    # ASR prefix stripping can corrupt structured outputs, e.g. extracting the
    # first quoted string from JSON. Format-AST validates the raw structure.
    return str(generation).strip()


def _balanced_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _extract_json_translation(generation: str) -> tuple[bool, str, str | None]:
    payload = _balanced_json_object(generation)
    if payload is None:
        return False, generation.strip(), "missing_json_object"
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        return False, generation.strip(), f"invalid_json:{exc.msg}"
    required = {"source_language", "target_language", "translation"}
    keys = set(parsed)
    if keys != required:
        return False, str(parsed.get("translation", "")).strip(), "json_keys_mismatch"
    translation = parsed.get("translation")
    if not isinstance(translation, str) or not translation.strip():
        return False, "", "missing_translation"
    return True, translation.strip(), None


def _extract_srt_translation(generation: str) -> tuple[bool, str, str | None]:
    lines = [line.strip() for line in generation.strip().splitlines() if line.strip()]
    if len(lines) < 3:
        return False, generation.strip(), "too_few_srt_lines"
    if lines[0] != "1":
        return False, " ".join(lines[2:]).strip(), "missing_srt_index"
    timestamp = r"^\d{2}:\d{2}:\d{2},\d{3}\s+-->\s+\d{2}:\d{2}:\d{2},\d{3}$"
    if re.match(timestamp, lines[1]) is None:
        return False, " ".join(lines[2:]).strip(), "invalid_srt_timestamp"
    translation = " ".join(lines[2:]).strip()
    if not translation:
        return False, "", "missing_translation"
    return True, translation, None


def _split_markdown_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def _extract_markdown_translation(generation: str) -> tuple[bool, str, str | None]:
    rows = [line.strip() for line in generation.strip().splitlines() if line.strip()]
    table_rows = [line for line in rows if "|" in line]
    if len(table_rows) < 3:
        return False, generation.strip(), "too_few_markdown_rows"
    header = [cell.lower() for cell in _split_markdown_row(table_rows[0])]
    expected = ["source_language", "target_language", "translation"]
    if header != expected:
        return False, generation.strip(), "markdown_header_mismatch"
    separator = _split_markdown_row(table_rows[1])
    if len(separator) != 3 or not all(re.fullmatch(r":?-{3,}:?", cell) for cell in separator):
        return False, generation.strip(), "markdown_separator_mismatch"
    data = _split_markdown_row(table_rows[2])
    if len(data) != 3:
        return False, generation.strip(), "markdown_data_mismatch"
    translation = data[2].strip()
    if not translation:
        return False, "", "missing_translation"
    return True, translation, None


def extract_formatted_translation(format_id: str, generation: str) -> tuple[bool, str, str | None]:
    if format_id == "json_object":
        return _extract_json_translation(generation)
    if format_id == "srt_single_cue":
        return _extract_srt_translation(generation)
    if format_id == "markdown_table":
        return _extract_markdown_translation(generation)
    return False, generation.strip(), f"unknown_format:{format_id}"


class NTTComplexEvaluator(BaseEvaluator):
    """Mixed evaluator for NTT-COMPLEX.

    `Format-AST` first validates and extracts the required structure, then
    scores the extracted translation with the standard audio translation path.
    """

    def __init__(self, config: dict[str, Any], num_parallel_requests=10):
        super().__init__(config, num_parallel_requests)
        self.audio_config = _audio_config(config)

    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        task_type = data_point.get("task_type")
        if task_type != "Format-AST":
            raise ValueError(f"Unsupported NTT-COMPLEX task_type: {task_type}")

        generation = _clean_generation(data_point.get("generation", ""), self.audio_config)
        format_id = str(data_point.get("format_id", ""))
        format_valid, extracted, format_error = extract_formatted_translation(format_id, generation)

        extra_fields = data_point.get("extra_fields") or {}
        tgt_lang = extra_fields.get("tgt_lang")
        translation_metrics = evaluate_translation(data_point.get("expected_answer", ""), extracted, tgt_lang)
        translation_ok = bool(translation_metrics.get("is_correct"))
        translation_metrics["is_correct"] = bool(format_valid and translation_ok)
        translation_metrics.update(
            {
                "format_valid": bool(format_valid),
                "format_score": 1.0 if format_valid else 0.0,
                "format_error": format_error,
                "formatted_generation": generation,
                "extracted_translation": extracted,
                "predicted_answer": extracted,
                "format_ast_is_correct": bool(format_valid and translation_ok),
            }
        )
        return translation_metrics
