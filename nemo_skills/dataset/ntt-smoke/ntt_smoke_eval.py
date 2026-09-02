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

"""Evaluator for NTT-SMOKE mixed audio/text manifests."""

from __future__ import annotations

import importlib.util
import re
from dataclasses import fields
from pathlib import Path
from typing import Any

from nemo_skills.evaluation.evaluator.audio import (
    AudioEvaluatorConfig,
    evaluate_sample as evaluate_audio_sample,
    extract_asr_text,
    strip_helpful_prefixes,
)
from nemo_skills.evaluation.evaluator.base import BaseEvaluator
from nemo_skills.evaluation.evaluator.contextasr import (
    calculate_wer,
    extract_entities,
    extract_entities_fuzzy,
    simple_tokenize,
)


_PREFERENCE_NORMALIZERS: dict[str, Any] = {}
SUCCESS_WER_THRESHOLD = 0.05


def _audio_config(config: dict[str, Any]) -> AudioEvaluatorConfig:
    """Build AudioEvaluatorConfig while ignoring NTT-specific config keys."""
    field_names = {field.name for field in fields(AudioEvaluatorConfig)}
    return AudioEvaluatorConfig(**{key: value for key, value in config.items() if key in field_names})


def _clean_generation(generation: str, config: AudioEvaluatorConfig) -> str:
    generation = extract_asr_text(str(generation).strip())
    if config.strip_helpful_prefixes:
        generation = strip_helpful_prefixes(generation)
    return generation.strip()


def _add_wer_correct_words(updates: dict[str, Any]) -> dict[str, Any]:
    if (
        "wer_ref_words" in updates
        and "wer_substitutions" in updates
        and "wer_deletions" in updates
    ):
        updates["wer_correct_words"] = max(
            0,
            int(updates["wer_ref_words"]) - int(updates["wer_substitutions"]) - int(updates["wer_deletions"]),
        )
    return updates


def _apply_wer_success_threshold(
    updates: dict[str, Any],
    threshold: float = SUCCESS_WER_THRESHOLD,
) -> dict[str, Any]:
    """Use NTT-SMOKE's stricter row-success threshold for WER tasks."""
    if updates.get("wer") is not None:
        threshold = float(threshold)
        updates["success_wer_threshold"] = threshold
        updates["success_wer_threshold_percent"] = round(100.0 * threshold, 6)
        updates["is_correct"] = float(updates["wer"]) < threshold
    return updates


def _load_preference_normalizer(normalizer_dir: str):
    normalizer_dir = str(Path(normalizer_dir))
    if normalizer_dir in _PREFERENCE_NORMALIZERS:
        return _PREFERENCE_NORMALIZERS[normalizer_dir]

    module_path = Path(normalizer_dir) / "preference_normalizer.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Preference-ASR normalizer not found: {module_path}")

    spec = importlib.util.spec_from_file_location(
        f"ntt_smoke_preference_normalizer_{abs(hash(normalizer_dir))}",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    normalizer = module.PreferenceAwareNormalizer()
    _PREFERENCE_NORMALIZERS[normalizer_dir] = normalizer
    return normalizer


def _wer_counts(ref: str, hyp: str) -> dict[str, Any]:
    import jiwer

    ref = str(ref).strip()
    hyp = str(hyp).strip()
    if not ref:
        hyp_words = hyp.split()
        insertions = len(hyp_words)
        return {
            "wer": 1.0 if insertions else 0.0,
            "wer_errors": insertions,
            "wer_ref_words": 0,
            "wer_substitutions": 0,
            "wer_insertions": insertions,
            "wer_deletions": 0,
            "wer_correct_words": 0,
        }

    measures = jiwer.process_words(ref, hyp)
    substitutions = measures.substitutions
    insertions = measures.insertions
    deletions = measures.deletions
    correct_words = measures.hits
    ref_words = substitutions + deletions + correct_words
    errors = substitutions + insertions + deletions
    wer = errors / ref_words if ref_words else 0.0
    return {
        "wer": wer,
        "wer_errors": errors,
        "wer_ref_words": ref_words,
        "wer_substitutions": substitutions,
        "wer_insertions": insertions,
        "wer_deletions": deletions,
        "wer_correct_words": correct_words,
    }


def _evaluate_preference_asr_sample(
    sample: dict[str, Any],
    generation: str,
    config: AudioEvaluatorConfig,
    success_wer_threshold: float = SUCCESS_WER_THRESHOLD,
) -> dict[str, Any]:
    normalizer_dir = (
        sample.get("preference_asr_normalizer_dir")
        or (str(Path(sample["preference_asr_dir"]) / "normalizer") if sample.get("preference_asr_dir") else None)
    )
    if not normalizer_dir:
        raise ValueError("PreferenceASR sample is missing preference_asr_normalizer_dir")

    normalizer = _load_preference_normalizer(str(normalizer_dir))
    cleaned_generation = _clean_generation(generation, config)
    reference = str(sample.get("expected_answer") or sample.get("preference_text") or "")
    norm_ref = normalizer.normalize_entry(reference, sample)
    norm_hyp = normalizer.normalize_entry(cleaned_generation, sample)
    if not str(norm_ref).strip() and reference.strip():
        norm_ref = reference.strip()
    updates = _wer_counts(norm_ref, norm_hyp)
    updates.update(
        {
            "text": norm_ref,
            "pred_text": norm_hyp,
            "predicted_answer": cleaned_generation,
        }
    )
    return _apply_wer_success_threshold(updates, success_wer_threshold)


def _evaluate_contextasr_sample(
    sample: dict[str, Any],
    generation: str,
    success_wer_threshold: float = SUCCESS_WER_THRESHOLD,
) -> dict[str, Any]:
    """Evaluate one ContextASR sample and retain edit-operation counts."""
    reference = sample["expected_answer"]
    entity_list = sample.get("entity_list") or []

    norm_ref = simple_tokenize(reference)
    norm_hyp = simple_tokenize(generation)
    ref_tokens = norm_ref.split()
    hyp_tokens = norm_hyp.split()

    wer, wer_i, wer_d, wer_s = calculate_wer(hyp_tokens, ref_tokens)
    wer_errors = wer_i + wer_d + wer_s
    wer_ref_words = len(ref_tokens)

    updates = {
        "wer": wer,
        "wer_errors": wer_errors,
        "wer_ref_words": wer_ref_words,
        "wer_insertions": wer_i,
        "wer_deletions": wer_d,
        "wer_substitutions": wer_s,
        "wer_correct_words": max(0, wer_ref_words - wer_s - wer_d),
        "text": norm_ref,
        "pred_text": norm_hyp,
        "predicted_answer": generation,
    }
    _apply_wer_success_threshold(updates, success_wer_threshold)

    norm_entities = []
    for entity in entity_list:
        norm_entity = simple_tokenize(entity)
        if norm_entity and norm_entity in norm_ref:
            norm_entities.append(norm_entity)

    if not norm_entities:
        updates.update(
            {
                "ne_wer": 0.0,
                "ne_wer_errors": 0,
                "ne_wer_ref_words": 0,
                "ne_fnr": 0.0,
                "ne_fnr_hits": 0,
                "ne_fnr_total": 0,
            }
        )
        return updates

    ref_entities = extract_entities(norm_ref, norm_entities)
    entity2count: dict[str, int] = {}
    for entity in ref_entities:
        entity2count[entity] = entity2count.get(entity, 0) + 1

    counted_entities = list(entity2count)
    hyp_exact_entities = extract_entities(norm_hyp, counted_entities, entity2count)
    hyp_fuzzy_entities = extract_entities_fuzzy(norm_hyp, counted_entities)

    ref_entity_tokens = " ".join(ref_entities).split()
    hyp_fuzzy_tokens = " ".join(hyp_fuzzy_entities).split()

    if ref_entity_tokens:
        ne_wer, ne_i, ne_d, ne_s = calculate_wer(hyp_fuzzy_tokens, ref_entity_tokens)
        ne_wer_errors = ne_i + ne_d + ne_s
        ne_wer_ref_words = len(ref_entity_tokens)
    else:
        ne_wer = 0.0
        ne_wer_errors = 0
        ne_wer_ref_words = 0

    ne_total = len(ref_entities)
    ne_hits = len(hyp_exact_entities)
    ne_fnr = 1.0 - (ne_hits / ne_total) if ne_total > 0 else 0.0

    updates.update(
        {
            "ne_wer": ne_wer,
            "ne_wer_errors": ne_wer_errors,
            "ne_wer_ref_words": ne_wer_ref_words,
            "ne_fnr": ne_fnr,
            "ne_fnr_hits": ne_hits,
            "ne_fnr_total": ne_total,
        }
    )
    return updates


def _extract_choice(text: str) -> str:
    """Extract an MCQ answer letter from a generation."""
    clean = str(text).strip()
    patterns = [
        r"(?:answer|option|choice)\s*(?:is|:)?\s*([A-J])\b",
        r"\b([A-J])\)",
        r"\b([A-J])\.",
        r"^\s*([A-J])\s*$",
        r"\b([A-J])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, clean, flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def _evaluate_text_mcq(sample: dict[str, Any]) -> dict[str, Any]:
    expected = str(sample.get("expected_answer", "")).strip().upper()
    generation = str(sample.get("generation", "")).strip()
    predicted = _extract_choice(generation)
    is_correct = bool(predicted) and predicted == expected
    return {
        "predicted_answer": predicted or generation,
        "is_correct": is_correct,
        "text": expected,
        "pred_text": predicted or generation,
    }


def _add_strict_hallucination(updates: dict[str, Any], generation: str) -> None:
    stripped = generation.strip()
    updates["strict_hallucination_rate"] = 1.0 if stripped else 0.0
    updates["nonempty_output_rate"] = 1.0 if stripped else 0.0
    updates["nonempty_chars"] = len(stripped)
    updates["predicted_answer"] = generation


class NTTSmokeEvaluator(BaseEvaluator):
    """Mixed evaluator for NTT-SMOKE.

    Most audio tasks reuse the standard audio evaluator. ContextASR samples
    reuse the ContextASR normalization/entity logic, and simple text samples
    use exact MCQ-letter scoring.
    """

    def __init__(self, config: dict[str, Any], num_parallel_requests=10):
        super().__init__(config, num_parallel_requests)
        self.audio_config = _audio_config(config)
        self.success_wer_threshold = float(config.get("success_wer_threshold", SUCCESS_WER_THRESHOLD))

    async def eval_single(self, data_point: dict[str, Any]) -> dict[str, Any]:
        task_type = data_point.get("task_type")

        if task_type == "ContextASR":
            generation = _clean_generation(data_point.get("generation", ""), self.audio_config)
            return _evaluate_contextasr_sample(data_point, generation, self.success_wer_threshold)

        if task_type == "Text-MCQ":
            return _evaluate_text_mcq(data_point)

        if task_type == "PreferenceASR":
            return _evaluate_preference_asr_sample(
                data_point,
                data_point.get("generation", ""),
                self.audio_config,
                self.success_wer_threshold,
            )

        updates = _add_wer_correct_words(evaluate_audio_sample(data_point, self.audio_config))
        _apply_wer_success_threshold(updates, self.success_wer_threshold)
        if task_type == "Hallucination":
            generation = _clean_generation(data_point.get("generation", ""), self.audio_config)
            _add_strict_hallucination(updates, generation)
        return updates
